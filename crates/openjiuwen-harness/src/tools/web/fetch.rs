//! `fetch_webpage`: HTTP GET through an injected fetcher, charset decoding and text extraction.
//!
//! Ports `openjiuwen/harness/tools/web/fetch_webpage.py` and its `_http.py`/`_decode.py`
//! collaborators as far as the Rust rewrite goes: connect through an [`HttpFetcher`] seam,
//! read the body under a byte cap, decode it with the detected charset and render HTML to
//! text through [`super::html`].
//!
//! Deviations from Python: no `aiohttp` proxy/pool plumbing (the `reqwest` client owns it), no
//! Jina-reader fallback, no web-research budget, no LLM quality/assessment pass and no
//! challenge detection. Every expected failure is a `ToolPayload::fail(...)`; `Err` is
//! reserved for cancellation and protocol violations.

// The registry contract fixes the `Fetch*` names inside this `fetch` module.
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
use super::html;
use crate::tools::descriptions;
use crate::tools::support::{
    parse_args, schema_for, tool_card, truncate_chars, ToolEnv, ToolPayload,
};

/// The model-facing tool name.
pub const TOOL_NAME: &str = "fetch_webpage";

/// Default request timeout, in seconds (Python `_FETCH_WEBPAGE_DEFAULT_TIMEOUT_SECONDS`).
pub const DEFAULT_TIMEOUT_SECONDS: u64 = 45;

/// Lower bound of the request timeout, in seconds.
pub const MIN_TIMEOUT_SECONDS: u64 = 5;

/// Upper bound of the request timeout, in seconds.
pub const MAX_TIMEOUT_SECONDS: u64 = 600;

/// Default cap of the extracted content, in characters (Python `_FETCH_WEBPAGE_DEFAULT_MAX_CHARS`).
pub const DEFAULT_MAX_CHARS: usize = 20_000;

/// Lower bound of a non-zero `max_chars` request.
pub const MIN_MAX_CHARS: usize = 500;

/// Default cap of the raw response body, in bytes (Python `_FETCH_WEBPAGE_DEFAULT_MAX_BYTES`).
pub const DEFAULT_MAX_BYTES: usize = 8_000_000;

/// Number of redirect hops `fetch_webpage` follows before giving up.
pub const REDIRECT_LIMIT: usize = 10;

/// The `User-Agent` sent by [`ReqwestFetcher`] (Python `_USER_AGENT`).
pub const DEFAULT_USER_AGENT: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
     AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36";

/// One HTTP GET handed to an [`HttpFetcher`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchRequest {
    /// The absolute `http`/`https` URL to fetch.
    pub url: String,
    /// The whole-request deadline, redirects included.
    pub timeout: Duration,
    /// The maximum number of body bytes the fetcher may read.
    pub max_bytes: usize,
    /// Extra request headers, sent in order.
    pub headers: Vec<(String, String)>,
}

/// The response an [`HttpFetcher`] returns.
///
/// The body is capped at [`FetchRequest::max_bytes`]; a body whose length equals the cap means
/// the fetcher stopped reading, which the tool resolves with the `Content-Length` header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchResponse {
    /// The HTTP status code.
    pub status: u16,
    /// The URL the response came from, after redirects.
    pub final_url: String,
    /// The response headers, in wire order.
    pub headers: Vec<(String, String)>,
    /// The (possibly capped) response body.
    pub body: Vec<u8>,
}

impl FetchResponse {
    /// Looks a header up case-insensitively.
    #[must_use]
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }

    /// The `Content-Type` header value, or the empty string when absent.
    #[must_use]
    pub fn content_type(&self) -> &str {
        self.header("content-type").unwrap_or("")
    }
}

/// The injected HTTP seam of `fetch_webpage`.
///
/// Implementations must return the transport-level response even for an error status (the tool
/// renders 4xx/5xx as payload failures), must never read more than
/// [`FetchRequest::max_bytes`], and must never follow a redirect: the tool resolves hops itself
/// so that the network guard runs on every one of them.
#[async_trait]
pub trait HttpFetcher: Send + Sync {
    /// Performs one GET request.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Timeout`] when the request deadline expires and [`Error::Transport`]
    /// for connection, TLS, redirect-policy and body-read failures.
    async fn fetch(&self, request: FetchRequest) -> Result<FetchResponse>;
}

/// The default [`HttpFetcher`], backed by `reqwest` over rustls.
///
/// Client construction happens once; a construction failure is stored and reported by the
/// first [`HttpFetcher::fetch`] call instead of panicking in the constructor.
pub struct ReqwestFetcher {
    /// The built client, or the construction error message.
    client: std::result::Result<reqwest::Client, String>,
}

impl ReqwestFetcher {
    /// Builds a fetcher with [`DEFAULT_USER_AGENT`].
    ///
    /// Redirects are never followed here: [`FetchWebpageTool`] follows them hop by hop so the
    /// network guard inspects every target.
    #[must_use]
    pub fn new() -> Self {
        Self::with_user_agent(DEFAULT_USER_AGENT)
    }

    /// Builds a fetcher with a custom `User-Agent`.
    #[must_use]
    pub fn with_user_agent(user_agent: impl Into<String>) -> Self {
        let client = reqwest::Client::builder()
            .user_agent(user_agent.into())
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|error| error.to_string());
        Self { client }
    }

    /// Wraps an already-configured `reqwest` client (proxy, TLS roots, connection pool).
    #[must_use]
    pub fn from_client(client: reqwest::Client) -> Self {
        Self { client: Ok(client) }
    }
}

impl Default for ReqwestFetcher {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl HttpFetcher for ReqwestFetcher {
    async fn fetch(&self, request: FetchRequest) -> Result<FetchResponse> {
        let client = self.client.as_ref().map_err(|message| {
            Error::transport(format!("failed to build the HTTP client: {message}"))
        })?;
        let mut builder = client.get(&request.url).timeout(request.timeout);
        for (name, value) in &request.headers {
            builder = builder.header(name.as_str(), value.as_str());
        }
        let mut response = builder
            .send()
            .await
            .map_err(|error| map_request_error(&error))?;
        let status = response.status().as_u16();
        let final_url = response.url().to_string();
        let headers = response
            .headers()
            .iter()
            .map(|(name, value)| {
                (
                    name.to_string(),
                    value.to_str().unwrap_or_default().to_owned(),
                )
            })
            .collect();
        let mut body = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|error| map_request_error(&error))?
        {
            let remaining = request.max_bytes.saturating_sub(body.len());
            if chunk.len() >= remaining {
                body.extend_from_slice(&chunk[..remaining]);
                break;
            }
            body.extend_from_slice(&chunk);
        }
        Ok(FetchResponse {
            status,
            final_url,
            headers,
            body,
        })
    }
}

/// Maps a `reqwest` failure onto the shared error type, keeping timeouts distinguishable.
fn map_request_error(error: &reqwest::Error) -> Error {
    if error.is_timeout() {
        Error::timeout(format!("request timed out: {error}"))
    } else if error.is_connect() {
        Error::transport(format!("connection failed: {error}"))
    } else {
        Error::transport(error.to_string())
    }
}

/// Arguments of the `fetch_webpage` tool.
///
/// Python parity: `url`, `max_chars` and `timeout_seconds` of `WebFetchWebpageTool.invoke`.
/// `max_bytes` is a Rust addition that bounds the raw body, and `prompt`/`caption` are accepted
/// only to stay tolerant of older callers: they are echoed back in the payload and never sent
/// to a model.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FetchWebpageArgs {
    /// The URL to fetch. A missing scheme defaults to `https`.
    pub url: String,
    /// Maximum characters of extracted content to return; `0` disables clipping.
    #[serde(default)]
    #[schemars(with = "u64")]
    pub max_chars: Option<u64>,
    /// Maximum response-body bytes to read before the download stops.
    #[serde(default)]
    #[schemars(with = "u64")]
    pub max_bytes: Option<u64>,
    /// Request timeout in seconds (clamped to 5..=600).
    #[serde(default)]
    #[schemars(with = "u64")]
    pub timeout_seconds: Option<u64>,
    /// Caller note echoed back in the payload; never sent to a model.
    #[serde(default)]
    pub prompt: Option<String>,
    /// Caller caption echoed back in the payload; never sent to a model.
    #[serde(default)]
    pub caption: Option<String>,
}

/// Fetches a URL and returns its title and plain text.
///
/// The tool resolves nothing against the workspace: a URL is not a filesystem path. The
/// permission pipeline still applies through [`ToolEnv::deny_reason`], which folds the network
/// guard for the fetch-tool aliases.
pub struct FetchWebpageTool {
    /// Path/permission environment (the workspace is unused).
    env: ToolEnv,
    /// The HTTP seam; `None` makes every call fail with a configuration message.
    fetcher: Option<Arc<dyn HttpFetcher>>,
    /// The model-facing card built from the constructor language.
    card: ToolCard,
}

impl FetchWebpageTool {
    /// Builds the tool over `env` with an optional [`HttpFetcher`].
    #[must_use]
    pub fn new(
        env: ToolEnv,
        fetcher: Option<Arc<dyn HttpFetcher>>,
        language: impl Into<String>,
    ) -> Self {
        let language = language.into();
        let card = Self::card_for(&language);
        Self {
            env: env.with_language(language),
            fetcher,
            card,
        }
    }

    /// Builds the model-facing card in `language`.
    ///
    /// The description text lives in [`descriptions`] and is empty until the table gains the
    /// `fetch_webpage` row; the card itself is always complete.
    #[must_use]
    pub fn card_for(language: &str) -> ToolCard {
        tool_card(
            TOOL_NAME,
            descriptions::description(TOOL_NAME, language),
            schema_for::<FetchWebpageArgs>(),
            true,
        )
    }

    /// Runs one fetch and renders the payload.
    async fn run(&self, inputs: &Value) -> ToolPayload {
        if let Some(reason) = self.env.deny_reason(TOOL_NAME, inputs) {
            return ToolPayload::fail(reason);
        }
        let args: FetchWebpageArgs = match parse_args(inputs) {
            Ok(args) => args,
            Err(payload) => return payload,
        };
        let url = match request_url(args.url.trim()) {
            Ok(url) => url,
            Err(message) => return ToolPayload::fail(message),
        };
        let Some(fetcher) = self.fetcher.as_ref() else {
            return ToolPayload::fail("http fetcher is not configured");
        };
        let deadline = tokio::time::Instant::now()
            + Duration::from_secs(resolve_timeout_seconds(args.timeout_seconds));
        match self
            .fetch_following_redirects(fetcher, &url, deadline, resolve_max_bytes(args.max_bytes))
            .await
        {
            Ok(response) => render_response(&url, &response, &args),
            Err(message) => ToolPayload::fail(message),
        }
    }

    /// Fetches `url`, following redirects hop by hop.
    ///
    /// The [`HttpFetcher`] never follows a redirect itself, so every hop is resolved here and
    /// passes through the same [`ToolEnv::deny_reason`] network guard as the initial URL: a
    /// public URL that redirects to a denied host (a loopback or link-local address, say) is
    /// refused before the request to it is made. At most [`REDIRECT_LIMIT`] hops are followed,
    /// and the whole chain shares one deadline.
    async fn fetch_following_redirects(
        &self,
        fetcher: &Arc<dyn HttpFetcher>,
        url: &str,
        deadline: tokio::time::Instant,
        max_bytes: usize,
    ) -> std::result::Result<FetchResponse, String> {
        let mut current = url.to_owned();
        for _hop in 0..=REDIRECT_LIMIT {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                return Err(format!(
                    "failed to fetch webpage: timeout: the request deadline expired before {current}"
                ));
            }
            let request = FetchRequest {
                url: current.clone(),
                timeout: remaining,
                max_bytes,
                headers: default_headers(),
            };
            let response = fetcher
                .fetch(request)
                .await
                .map_err(|error| format!("failed to fetch webpage: {error}"))?;
            let Some(location) = redirect_target(&response) else {
                return Ok(response);
            };
            let next = redirect_url(&current, location)?;
            let host = url_host(&next).to_owned();
            if let Some(reason) = self.env.deny_reason(TOOL_NAME, &json!({ "url": next })) {
                return Err(format!(
                    "refusing to follow a redirect to '{host}': {reason}"
                ));
            }
            current = next;
        }
        Err(format!(
            "too many redirects (more than {REDIRECT_LIMIT}) fetching {url}"
        ))
    }
}

/// The `Location` of a redirect response, if `response` is one.
fn redirect_target(response: &FetchResponse) -> Option<&str> {
    if !matches!(response.status, 301 | 302 | 303 | 307 | 308) {
        return None;
    }
    response
        .header("location")
        .map(str::trim)
        .filter(|location| !location.is_empty())
}

/// Resolves the `Location` of a redirect against the URL that produced it.
///
/// # Errors
///
/// Returns a failure payload when the location cannot be resolved into an absolute http(s) URL.
fn redirect_url(current: &str, location: &str) -> std::result::Result<String, String> {
    let resolved = html::normalize_url(current, location).ok_or_else(|| {
        format!("redirect from '{current}' has an unusable Location '{location}'")
    })?;
    request_url(&resolved)
        .map_err(|message| format!("redirect from '{current}' is unusable: {message}"))
}

/// The authority (`host[:port]`) of an absolute URL.
fn url_host(url: &str) -> &str {
    let rest = url.split_once("://").map_or(url, |(_, rest)| rest);
    let authority = rest.split(['/', '?', '#']).next().unwrap_or(rest);
    authority.rsplit('@').next().unwrap_or(authority)
}

/// Turns a fetched response into the tool payload.
fn render_response(url: &str, response: &FetchResponse, args: &FetchWebpageArgs) -> ToolPayload {
    let content_type = response.content_type().to_owned();
    let media = media_type(&content_type);
    let charset = detect_charset(&content_type, &response.body);
    let decoded = decode_body(&response.body, &charset);
    if !(200..300).contains(&response.status) {
        let excerpt = failure_excerpt(&media, &decoded);
        let mut message = format!("HTTP {} fetching {url}", response.status);
        if !excerpt.is_empty() {
            message.push_str(": ");
            message.push_str(&excerpt);
        }
        return ToolPayload::fail_with_data(
            message,
            json!({
                "url": url,
                "final_url": response.final_url,
                "status": response.status,
                "content_type": content_type,
            }),
        );
    }
    let rendered = match render_body(&media, &decoded) {
        Ok(rendered) => rendered,
        Err(message) => return ToolPayload::fail(message),
    };
    let clipped = truncate_chars(&rendered.content, resolve_max_chars(args.max_chars));
    let truncated =
        clipped.truncated || body_was_truncated(response, resolve_max_bytes(args.max_bytes));
    let mut data = json!({
        "url": url,
        "final_url": response.final_url,
        "status": response.status,
        "content_type": content_type,
        "charset": charset,
        "content": clipped.text,
        "truncated": truncated,
    });
    if let Some(title) = rendered.title {
        data["title"] = Value::String(title);
    }
    if let Some(prompt) = args.prompt.as_ref() {
        data["prompt"] = Value::String(prompt.clone());
    }
    if let Some(caption) = args.caption.as_ref() {
        data["caption"] = Value::String(caption.clone());
    }
    ToolPayload::ok(data)
}

#[async_trait]
impl Tool for FetchWebpageTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
        ctx.check_cancelled()?;
        Ok(self.run(&inputs).await.to_json())
    }
}

/// The rendered body of a successful response.
struct RenderedBody {
    /// Text handed to the model.
    content: String,
    /// The document title, for HTML responses.
    title: Option<String>,
}

/// Renders a decoded body for `media` (the lowercased media type of the response).
fn render_body(media: &str, decoded: &str) -> std::result::Result<RenderedBody, String> {
    if media == "application/json" || media.ends_with("+json") {
        return Ok(RenderedBody {
            content: pretty_json(decoded),
            title: None,
        });
    }
    let html_like = media == "text/html"
        || media == "application/xhtml+xml"
        || (media.is_empty() && looks_like_html(decoded));
    if html_like {
        let extracted = html::extract(decoded);
        return Ok(RenderedBody {
            content: extracted.text,
            title: extracted.title,
        });
    }
    if media.is_empty() || media.starts_with("text/") || is_xml_or_script(media) {
        return Ok(RenderedBody {
            content: decoded.to_owned(),
            title: None,
        });
    }
    Err(format!(
        "unsupported content type '{media}': fetch_webpage only renders text, HTML and JSON responses"
    ))
}

/// Whether `media` is an XML or JavaScript text type.
fn is_xml_or_script(media: &str) -> bool {
    matches!(
        media,
        "application/xml" | "application/javascript" | "application/x-javascript"
    ) || media.ends_with("+xml")
}

/// Extracts the lowercased media type from a `Content-Type` value.
fn media_type(content_type: &str) -> String {
    content_type
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase()
}

/// Pretty-prints `text` when it is valid JSON, and passes it through unchanged otherwise.
fn pretty_json(text: &str) -> String {
    match serde_json::from_str::<Value>(text) {
        Ok(value) => serde_json::to_string_pretty(&value).unwrap_or_else(|_| text.to_owned()),
        Err(_) => text.to_owned(),
    }
}

/// Whether an unlabelled body looks like HTML.
fn looks_like_html(text: &str) -> bool {
    let head: String = text
        .chars()
        .take(1024)
        .collect::<String>()
        .to_ascii_lowercase();
    head.contains("<html") || head.contains("<!doctype html") || head.contains("<body")
}

/// Collapses the first 200 characters of a failure body into one line.
fn excerpt(text: &str) -> String {
    let collapsed: String = text
        .chars()
        .take(200)
        .map(|character| {
            if character.is_whitespace() {
                ' '
            } else {
                character
            }
        })
        .collect();
    collapsed.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Renders a one-line excerpt of an error response body, stripping markup when it is HTML.
fn failure_excerpt(media: &str, decoded: &str) -> String {
    let text =
        if media == "text/html" || media == "application/xhtml+xml" || looks_like_html(decoded) {
            html::extract(decoded).text
        } else {
            decoded.to_owned()
        };
    excerpt(&text)
}

/// Validates and normalizes a model-supplied URL.
///
/// A blank URL, a control character, whitespace or a non-http(s) scheme is rejected; a URL
/// without a scheme gets `https://`. A dotted prefix followed by a port
/// (`example.com:8080/x`) is treated as a host, not as a scheme.
fn request_url(raw: &str) -> std::result::Result<String, String> {
    if raw.is_empty() {
        return Err("url cannot be empty".to_owned());
    }
    if raw.chars().any(char::is_control) {
        return Err(format!(
            "invalid URL '{raw}': it contains a control character"
        ));
    }
    if raw.chars().any(char::is_whitespace) {
        return Err(format!("invalid URL '{raw}': it contains whitespace"));
    }
    let absolute = match html::url_scheme(raw) {
        Some(scheme) if !scheme.contains('.') => {
            let lower = scheme.to_ascii_lowercase();
            if lower != "http" && lower != "https" {
                return Err(format!(
                    "unsupported URL scheme '{scheme}': only http and https are supported"
                ));
            }
            raw.to_owned()
        }
        _ => format!("https://{raw}"),
    };
    html::normalize_url(&absolute, &absolute).ok_or_else(|| format!("invalid URL '{raw}'"))
}

/// Clamps the requested content cap.
fn resolve_max_chars(value: Option<u64>) -> usize {
    match value {
        None => DEFAULT_MAX_CHARS,
        Some(0) => 0,
        Some(value) => usize::try_from(value)
            .unwrap_or(DEFAULT_MAX_CHARS)
            .max(MIN_MAX_CHARS),
    }
}

/// Clamps the requested body cap to `1..=DEFAULT_MAX_BYTES`.
fn resolve_max_bytes(value: Option<u64>) -> usize {
    let value = value.map_or(DEFAULT_MAX_BYTES, |value| {
        usize::try_from(value).unwrap_or(DEFAULT_MAX_BYTES)
    });
    value.clamp(1, DEFAULT_MAX_BYTES)
}

/// Clamps the requested timeout to `MIN_TIMEOUT_SECONDS..=MAX_TIMEOUT_SECONDS`.
fn resolve_timeout_seconds(value: Option<u64>) -> u64 {
    value
        .unwrap_or(DEFAULT_TIMEOUT_SECONDS)
        .clamp(MIN_TIMEOUT_SECONDS, MAX_TIMEOUT_SECONDS)
}

/// The headers every fetch sends.
fn default_headers() -> Vec<(String, String)> {
    vec![
        ("User-Agent".to_owned(), DEFAULT_USER_AGENT.to_owned()),
        (
            "Accept".to_owned(),
            "text/html,application/xhtml+xml,application/json;q=0.9,text/plain;q=0.8,*/*;q=0.5"
                .to_owned(),
        ),
    ]
}

/// Whether the body was cut off by the byte cap rather than by the end of the document.
fn body_was_truncated(response: &FetchResponse, max_bytes: usize) -> bool {
    if response.body.len() < max_bytes {
        return false;
    }
    match response
        .header("content-length")
        .and_then(|value| value.trim().parse::<usize>().ok())
    {
        Some(length) => length > response.body.len(),
        None => true,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::collections::BTreeMap;

    use tempfile::TempDir;

    use super::*;
    use crate::permission::{
        PermissionEngine, PermissionEngineConfig, PermissionLevel, PermissionsSection,
    };
    use crate::tools::web::test_support::{fetch_response as response, StubFetcher};
    use crate::workspace::Workspace;

    /// Builds a tool over a fresh temp workspace.
    fn tool(fetcher: Option<Arc<dyn HttpFetcher>>, dir: &TempDir) -> FetchWebpageTool {
        FetchWebpageTool::new(ToolEnv::new(Workspace::new(dir.path())), fetcher, "en")
    }

    /// Builds a tool whose permission engine denies `fetch_webpage`.
    fn denied_tool(fetcher: Arc<dyn HttpFetcher>, dir: &TempDir) -> FetchWebpageTool {
        let section = PermissionsSection {
            package_builtin_rules: false,
            defaults: BTreeMap::from([("*".to_owned(), PermissionLevel::Allow)]),
            tools: BTreeMap::from([("mcp_fetch_webpage".to_owned(), PermissionLevel::Deny)]),
            ..PermissionsSection::default()
        };
        let engine = Arc::new(PermissionEngine::new(PermissionEngineConfig::new(section)));
        let env = ToolEnv::new(Workspace::new(dir.path())).with_permission(engine);
        FetchWebpageTool::new(env, Some(fetcher), "en")
    }

    /// Invokes the tool and returns the payload envelope.
    async fn invoke(tool: &FetchWebpageTool, inputs: Value) -> Value {
        tool.invoke(inputs, &ToolContext::new()).await.unwrap()
    }

    #[test]
    fn card_exposes_python_arguments() {
        let card = FetchWebpageTool::card_for("en");
        assert_eq!(card.name, TOOL_NAME);
        assert!(card.parallel_safe);
        assert_eq!(card.input_params["required"], json!(["url"]));
        let properties = card.input_params["properties"].as_object().unwrap();
        let mut keys: Vec<&str> = properties.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "caption",
                "max_bytes",
                "max_chars",
                "prompt",
                "timeout_seconds",
                "url"
            ]
        );
        assert_eq!(properties["max_chars"]["type"], json!("integer"));
        assert_eq!(properties["timeout_seconds"]["type"], json!("integer"));
    }

    #[test]
    fn clamps_the_requested_limits() {
        assert_eq!(resolve_max_chars(None), DEFAULT_MAX_CHARS);
        assert_eq!(resolve_max_chars(Some(0)), 0);
        assert_eq!(resolve_max_chars(Some(10)), MIN_MAX_CHARS);
        assert_eq!(resolve_max_chars(Some(25_000)), 25_000);

        assert_eq!(resolve_timeout_seconds(None), DEFAULT_TIMEOUT_SECONDS);
        assert_eq!(resolve_timeout_seconds(Some(1)), MIN_TIMEOUT_SECONDS);
        assert_eq!(resolve_timeout_seconds(Some(10_000)), MAX_TIMEOUT_SECONDS);
        assert_eq!(resolve_timeout_seconds(Some(60)), 60);

        assert_eq!(resolve_max_bytes(None), DEFAULT_MAX_BYTES);
        assert_eq!(resolve_max_bytes(Some(0)), 1);
        assert_eq!(resolve_max_bytes(Some(100)), 100);
        assert_eq!(resolve_max_bytes(Some(u64::MAX)), DEFAULT_MAX_BYTES);
    }

    #[test]
    fn validates_urls_and_their_schemes() {
        assert_eq!(
            request_url("example.com/a?b=1").unwrap(),
            "https://example.com/a?b=1"
        );
        assert_eq!(
            request_url("HTTPS://Example.COM:8443/a#frag").unwrap(),
            "https://example.com:8443/a"
        );
        assert_eq!(request_url("").unwrap_err(), "url cannot be empty");
        assert_eq!(
            request_url("file:///etc/passwd").unwrap_err(),
            "unsupported URL scheme 'file': only http and https are supported"
        );
        assert_eq!(
            request_url("mailto:a@b.example").unwrap_err(),
            "unsupported URL scheme 'mailto': only http and https are supported"
        );
        assert!(request_url("https://exa mple.com")
            .unwrap_err()
            .contains("whitespace"));
        assert_eq!(
            request_url("javascript:alert(1)").unwrap_err(),
            "unsupported URL scheme 'javascript': only http and https are supported"
        );
    }

    #[tokio::test]
    async fn renders_html_and_keeps_the_title() {
        let dir = tempfile::tempdir().unwrap();
        let fetcher = StubFetcher::ok(
            "text/html; charset=utf-8",
            "<html><head><title>Doc</title><style>b{}</style></head>\
             <body><h1>Heading</h1><p>Body text</p><script>x()</script></body></html>",
        );
        let payload = invoke(
            &tool(Some(fetcher), &dir),
            json!({"url": "https://example.com/"}),
        )
        .await;
        assert_eq!(payload["success"], json!(true));
        let data = &payload["data"];
        assert_eq!(data["title"], json!("Doc"));
        assert_eq!(data["content"], json!("Heading\nBody text"));
        assert_eq!(data["charset"], json!("utf-8"));
        assert_eq!(data["status"], json!(200));
        assert_eq!(data["final_url"], json!("https://example.com/final"));
        assert_eq!(data["truncated"], json!(false));
        assert!(data.get("prompt").is_none());
    }

    #[tokio::test]
    async fn renders_json_as_pretty_text() {
        let dir = tempfile::tempdir().unwrap();
        let fetcher = StubFetcher::ok("application/json", r#"{"name":"value","count":2}"#);
        let payload = invoke(
            &tool(Some(fetcher), &dir),
            json!({"url": "https://e.com/api"}),
        )
        .await;
        let content = payload["data"]["content"].as_str().unwrap();
        assert!(
            content.contains("\"name\": \"value\""),
            "content: {content}"
        );
        assert!(content.contains("\"count\": 2"), "content: {content}");
        assert!(payload["data"].get("title").is_none());
    }

    #[tokio::test]
    async fn passes_plain_text_and_unlabelled_html_through() {
        let dir = tempfile::tempdir().unwrap();
        let plain = StubFetcher::ok("text/plain", "  raw text  ");
        let payload = invoke(
            &tool(Some(plain), &dir),
            json!({"url": "https://e.com/t.txt"}),
        )
        .await;
        assert_eq!(payload["data"]["content"], json!("  raw text  "));

        let html = StubFetcher::ok(
            "",
            "<html><head><title>T</title></head><body><p>hi</p></body></html>",
        );
        let payload = invoke(&tool(Some(html), &dir), json!({"url": "https://e.com/h"})).await;
        assert_eq!(payload["data"]["title"], json!("T"));
        assert_eq!(payload["data"]["content"], json!("hi"));
    }

    #[tokio::test]
    async fn rejects_binary_content_types() {
        let dir = tempfile::tempdir().unwrap();
        let fetcher = StubFetcher::ok("application/pdf", "%PDF-1.7");
        let payload = invoke(
            &tool(Some(fetcher), &dir),
            json!({"url": "https://e.com/a.pdf"}),
        )
        .await;
        assert_eq!(payload["success"], json!(false));
        assert_eq!(
            payload["error"],
            json!(
                "unsupported content type 'application/pdf': fetch_webpage only renders text, HTML and JSON responses"
            )
        );
    }

    #[tokio::test]
    async fn decodes_the_header_charset() {
        let dir = tempfile::tempdir().unwrap();
        let fetcher = StubFetcher::ok("text/plain; charset=iso-8859-1", b"caf\xe9".to_vec());
        let payload = invoke(
            &tool(Some(fetcher), &dir),
            json!({"url": "https://e.com/l"}),
        )
        .await;
        assert_eq!(payload["data"]["charset"], json!("iso-8859-1"));
        assert_eq!(payload["data"]["content"], json!("café"));
    }

    #[tokio::test]
    async fn http_errors_become_failure_payloads() {
        let dir = tempfile::tempdir().unwrap();
        let fetcher = StubFetcher::new(Ok(response(404, "text/html", "<p>not found</p>")));
        let payload = invoke(
            &tool(Some(fetcher), &dir),
            json!({"url": "https://e.com/missing"}),
        )
        .await;
        assert_eq!(payload["success"], json!(false));
        assert_eq!(
            payload["error"],
            json!("HTTP 404 fetching https://e.com/missing: not found")
        );
        assert_eq!(payload["data"]["status"], json!(404));
    }

    #[tokio::test]
    async fn transport_errors_become_failure_payloads() {
        let dir = tempfile::tempdir().unwrap();
        let fetcher = StubFetcher::new(Err(Error::timeout("request timed out: slow")));
        let payload = invoke(
            &tool(Some(fetcher), &dir),
            json!({"url": "https://e.com/slow"}),
        )
        .await;
        assert_eq!(payload["success"], json!(false));
        assert_eq!(
            payload["error"],
            json!("failed to fetch webpage: timeout: request timed out: slow")
        );
    }

    #[tokio::test]
    async fn missing_fetcher_and_bad_arguments_fail() {
        let dir = tempfile::tempdir().unwrap();
        let payload = invoke(&tool(None, &dir), json!({"url": "https://e.com/"})).await;
        assert_eq!(payload["error"], json!("http fetcher is not configured"));

        let fetcher = StubFetcher::ok("text/plain", "x");
        let payload = invoke(&tool(Some(fetcher), &dir), json!({"url": "   "})).await;
        assert_eq!(payload["error"], json!("url cannot be empty"));

        let fetcher = StubFetcher::ok("text/plain", "x");
        let payload = invoke(&tool(Some(fetcher), &dir), json!({"url": 5})).await;
        assert!(payload["error"]
            .as_str()
            .unwrap()
            .starts_with("invalid tool arguments: "));

        let fetcher = StubFetcher::ok("text/plain", "x");
        let payload = invoke(&tool(Some(fetcher), &dir), json!(["not", "object"])).await;
        assert_eq!(payload["error"], json!("tool arguments must be an object"));
    }

    #[tokio::test]
    async fn permission_denial_never_reaches_the_fetcher() {
        let dir = tempfile::tempdir().unwrap();
        let fetcher = StubFetcher::ok("text/plain", "secret");
        let payload = invoke(
            &denied_tool(fetcher.clone(), &dir),
            json!({"url": "https://blocked.example/"}),
        )
        .await;
        assert_eq!(payload["success"], json!(false));
        assert!(payload["error"]
            .as_str()
            .unwrap()
            .contains("Denied by rule"));
        assert_eq!(fetcher.calls(), 0);
    }

    #[tokio::test]
    async fn truncates_the_content_and_reports_it() {
        let dir = tempfile::tempdir().unwrap();
        let body = "a".repeat(600);
        let fetcher = StubFetcher::ok("text/plain", body);
        let payload = invoke(
            &tool(Some(fetcher), &dir),
            json!({"url": "https://e.com/long", "max_chars": 500}),
        )
        .await;
        assert_eq!(payload["data"]["truncated"], json!(true));
        let content = payload["data"]["content"].as_str().unwrap();
        assert!(content.contains("output truncated"), "content: {content}");
        assert!(content.starts_with(&"a".repeat(500)));
    }

    #[tokio::test]
    async fn reports_a_body_cut_off_by_the_byte_cap() {
        let dir = tempfile::tempdir().unwrap();
        let fetcher = StubFetcher::ok("text/plain", "0123456789");
        let payload = invoke(
            &tool(Some(fetcher), &dir),
            json!({"url": "https://e.com/capped", "max_bytes": 10, "max_chars": 0}),
        )
        .await;
        assert_eq!(payload["data"]["content"], json!("0123456789"));
        assert_eq!(payload["data"]["truncated"], json!(true));
    }

    #[tokio::test]
    async fn echoes_prompt_and_caption_without_sending_them() {
        let dir = tempfile::tempdir().unwrap();
        let fetcher = StubFetcher::ok("text/plain", "text");
        let payload = invoke(
            &tool(Some(fetcher.clone()), &dir),
            json!({
                "url": "https://e.com/x",
                "prompt": "summarize",
                "caption": "fig 1",
                "max_chars": 1000,
            }),
        )
        .await;
        assert_eq!(payload["data"]["prompt"], json!("summarize"));
        assert_eq!(payload["data"]["caption"], json!("fig 1"));
        let request = fetcher.last_request().unwrap();
        assert_eq!(request.url, "https://e.com/x");
        // The redirect chain shares one deadline, so the first hop receives what is left of it.
        assert!(request.timeout <= Duration::from_secs(DEFAULT_TIMEOUT_SECONDS));
        assert!(request.timeout > Duration::from_secs(DEFAULT_TIMEOUT_SECONDS - 1));
        assert_eq!(request.max_bytes, DEFAULT_MAX_BYTES);
        assert!(request
            .headers
            .iter()
            .any(|(name, value)| name == "User-Agent" && value == DEFAULT_USER_AGENT));
    }

    #[test]
    fn pretty_json_only_rewrites_valid_documents() {
        assert_eq!(pretty_json("not json"), "not json");
        assert_eq!(pretty_json("[]"), "[]");
    }

    /// Keeps the argument schema exercised from this module.
    #[test]
    fn args_schema_is_an_object() {
        let schema = schema_for::<FetchWebpageArgs>();
        assert_eq!(schema["type"], json!("object"));
        assert_eq!(schema["required"], json!(["url"]));
    }
}
