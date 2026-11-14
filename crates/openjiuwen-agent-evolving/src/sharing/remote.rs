//! The remote experience-hub client: an HTTP [`SharingBackend`].
//!
//! The Python reference's `sharing` package is local-filesystem only, so this
//! transport is **new design, not a port**: the hub protocol below is defined by
//! this module and is the only description of it in the workspace.
//!
//! ## Hub protocol v1
//!
//! Every URL is `{base_url}` plus the path, where `{base_url}` has its trailing
//! `/` removed by [`RemoteHubOptions::validate`]. Bodies are JSON with
//! `snake_case` keys; unknown reply fields are ignored.
//!
//! | Method | Path | Request body | Success reply |
//! | --- | --- | --- | --- |
//! | `POST` | `/v1/hub/bundles` | the [`SharedSkillBundle`] JSON | `{"bundle_id": "..."}` |
//! | `GET` | `/v1/hub/bundles` | — | `{"bundles": [<bundle>, ...]}` |
//! | `GET` | `/v1/hub/bundles/{bundle_id}` | — | the bundle JSON |
//! | `DELETE` | `/v1/hub/bundles/{bundle_id}` | — | `{"removed": true}` |
//! | `POST` | `/v1/hub/bundles/search` | `{"query": <QueryKeywords>, "limit": n}` | `{"results": [<SkillSearchResult>, ...]}` |
//! | `GET` | `/health` | — | any `2xx`; the body is ignored |
//!
//! `GET` and `DELETE` on a bundle answer `404` when the id is unknown; that is a
//! normal outcome (`Ok(None)` / `Ok(false)`), never an error.
//!
//! ## Authentication
//!
//! When [`RemoteHubOptions::api_key`] is set the client sends
//! `Authorization: Bearer <key>` on every request. Static
//! [`headers`](RemoteHubOptions::headers) are attached to every request too, and
//! a static `Authorization` header is overridden by the bearer token.
//!
//! ## Error mapping
//!
//! - Connection failures, and any non-4xx non-2xx status (including a `5xx`
//!   whose retry budget is spent), map to [`Error::transport`].
//! - A request deadline maps to [`Error::timeout`]; the shared
//!   [`HttpClient`](openjiuwen_common::http::HttpClient) already draws that line.
//! - A `4xx` (including a `429` whose retry budget is spent) maps to
//!   [`Error::validation`], quoting the hub envelope `{"error":{"code",
//!   "message"}}` when the body carries one.
//! - A body that is not the expected JSON maps to [`Error::validation`] naming
//!   the endpoint.
//!
//! Retries cover `429`/`5xx` only, with [`RemoteHubOptions::max_retries`]
//! additional attempts after the first one; `4xx` is never retried.
//!
//! ## Client-side normalisation
//!
//! A sloppy hub still behaves like the local backends: [`list`](SharingBackend::list)
//! sorts by `bundle_id`, [`search`](SharingBackend::search) drops `0.0` scores,
//! sorts by descending score then ascending `bundle_id`, applies `limit` itself
//! and short-circuits an empty query without a request, and
//! [`publish`](SharingBackend::publish) falls back to the bundle's content-derived
//! id when the hub echoes a blank one.

use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_common::config::env;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::{HttpClient, HttpClientConfig, HttpRequest, HttpResponse};
use openjiuwen_common::security::{SecretString, REDACTED};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::sharing::backend::SharingBackend;
use crate::sharing::types::{QueryKeywords, SharedSkillBundle, SkillSearchResult};

/// Environment variable naming the hub base URL: `OPENJIUWEN_HUB_URL`.
pub const HUB_URL_ENV: &str = "OPENJIUWEN_HUB_URL";
/// Environment variable carrying the hub API key: `OPENJIUWEN_HUB_API_KEY`.
pub const HUB_API_KEY_ENV: &str = "OPENJIUWEN_HUB_API_KEY";

/// The default total timeout of one hub request.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
/// The default number of retries after the first attempt.
const DEFAULT_MAX_RETRIES: u32 = 2;
/// The backoff before the first retry.
const RETRY_INITIAL_BACKOFF: Duration = Duration::from_millis(50);
/// The upper bound on the retry backoff.
const RETRY_MAX_BACKOFF: Duration = Duration::from_millis(500);
/// The statuses a hub request is retried on: rate limiting and server errors.
const RETRY_STATUSES: [u16; 5] = [429, 500, 502, 503, 504];

/// The bundle collection path.
const BUNDLES_PATH: &str = "/v1/hub/bundles";
/// The keyword-search path.
const SEARCH_PATH: &str = "/v1/hub/bundles/search";
/// The liveness path.
const HEALTH_PATH: &str = "/health";

/// The endpoint label used in errors for [`SharingBackend::publish`].
const PUBLISH_ENDPOINT: &str = "POST /v1/hub/bundles";
/// The endpoint label used in errors for [`SharingBackend::list`].
const LIST_ENDPOINT: &str = "GET /v1/hub/bundles";
/// The endpoint label used in errors for [`SharingBackend::get`].
const GET_ENDPOINT: &str = "GET /v1/hub/bundles/{bundle_id}";
/// The endpoint label used in errors for [`SharingBackend::remove`].
const REMOVE_ENDPOINT: &str = "DELETE /v1/hub/bundles/{bundle_id}";
/// The endpoint label used in errors for [`SharingBackend::search`].
const SEARCH_ENDPOINT: &str = "POST /v1/hub/bundles/search";
/// The endpoint label used in errors for [`RemoteHubSharingBackend::health`].
const HEALTH_ENDPOINT: &str = "GET /health";

/// The longest unparsed error body quoted in an error message.
const ERROR_BODY_CHARS: usize = 200;

/// Connection settings for a remote experience hub.
///
/// The type holds one credential — [`api_key`](Self::api_key) — so its [`Debug`]
/// implementation redacts that key and prints header *names* only: a static
/// header may carry a credential of its own.
#[derive(Clone, PartialEq, Eq)]
pub struct RemoteHubOptions {
    /// The hub root, for example `https://hub.example.com`; [`Self::validate`]
    /// removes a trailing `/`.
    pub base_url: String,
    /// The bearer token sent as `Authorization: Bearer <key>` when present.
    pub api_key: Option<String>,
    /// Extra headers attached to every request, keyed by header name.
    pub headers: BTreeMap<String, String>,
    /// The total timeout of one request, including reading the body.
    pub timeout: Duration,
    /// The number of retries after the first attempt; `0` disables retries.
    pub max_retries: u32,
}

impl RemoteHubOptions {
    /// Creates options for `base_url` with no key, no extra headers, a 30-second
    /// timeout and two retries.
    ///
    /// The URL is validated by [`Self::validate`], which
    /// [`RemoteHubSharingBackend::new`] calls.
    #[must_use]
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: None,
            headers: BTreeMap::new(),
            timeout: DEFAULT_TIMEOUT,
            max_retries: DEFAULT_MAX_RETRIES,
        }
    }

    /// Sets the bearer token used for every request.
    #[must_use]
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Adds one header sent with every request.
    #[must_use]
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }

    /// Adds several headers sent with every request.
    #[must_use]
    pub fn with_headers(mut self, headers: impl IntoIterator<Item = (String, String)>) -> Self {
        self.headers.extend(headers);
        self
    }

    /// Sets the total timeout of one request.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Sets the number of retries after the first attempt.
    #[must_use]
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Reads the hub configuration from the process environment.
    ///
    /// `OPENJIUWEN_HUB_URL` is required and names the base URL; the optional
    /// `OPENJIUWEN_HUB_API_KEY` becomes the bearer token. Blank values count as
    /// unset. Every other field keeps its [`Self::new`] default.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when `OPENJIUWEN_HUB_URL` is unset or blank, or
    /// when the base URL fails [`Self::validate`].
    pub fn from_env() -> Result<Self> {
        let base_url = env::var(HUB_URL_ENV).ok_or_else(|| {
            Error::validation(format!(
                "{HUB_URL_ENV} is not set: no remote sharing hub is configured"
            ))
        })?;
        let mut options = Self::new(base_url);
        if let Some(api_key) = env::var(HUB_API_KEY_ENV) {
            options.api_key = Some(api_key);
        }
        options.validate()?;
        Ok(options)
    }

    /// Checks the base URL and normalizes it in place.
    ///
    /// The trimmed URL must use the `http` or `https` scheme and carry a
    /// non-empty host; trailing `/` characters are removed so
    /// `https://hub.example.com/` and `https://hub.example.com` address the same
    /// hub, path prefix included. The host check is a shape check, not full URL
    /// parsing: header and key values are validated by the HTTP layer on send.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] naming the offending base URL when it is empty,
    /// uses another scheme, or has no host.
    pub fn validate(&mut self) -> Result<()> {
        self.base_url = normalise_base_url(&self.base_url)?;
        Ok(())
    }
}

impl fmt::Debug for RemoteHubOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RemoteHubOptions")
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key.as_ref().map(|_| REDACTED))
            .field("headers", &self.headers.keys().collect::<Vec<&String>>())
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .finish()
    }
}

/// A [`SharingBackend`] that stores bundles on a remote experience hub.
///
/// The client is cheap to clone and safe to share; construct one per hub and
/// reuse it. [`RemoteHubSharingBackend::new`] validates the options once, so a
/// bad base URL is rejected before any request is attempted.
#[derive(Debug, Clone)]
pub struct RemoteHubSharingBackend {
    /// The shared transport, configured with the options' timeout and retries.
    client: HttpClient,
    /// The validated hub settings.
    options: RemoteHubOptions,
}

impl RemoteHubSharingBackend {
    /// Builds a backend from `options`, validating and normalizing them.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when the base URL is invalid, and the errors of
    /// [`HttpClient::new`](openjiuwen_common::http::HttpClient::new) when the
    /// transport cannot be built.
    pub fn new(mut options: RemoteHubOptions) -> Result<Self> {
        options.validate()?;
        let config = HttpClientConfig::default()
            .with_timeout(options.timeout)
            .with_max_retries(options.max_retries)
            .with_retry_backoff(RETRY_INITIAL_BACKOFF, RETRY_MAX_BACKOFF)
            .with_retry_statuses(RETRY_STATUSES)
            .with_user_agent(format!("{}/{}", crate::CRATE_NAME, crate::CRATE_VERSION));
        Ok(Self {
            client: HttpClient::new(config)?,
            options,
        })
    }

    /// Builds a backend pointing at `url` with the default options.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::new`].
    pub fn from_url(url: impl Into<String>) -> Result<Self> {
        Self::new(RemoteHubOptions::new(url))
    }

    /// The validated options this backend was built with.
    #[must_use]
    pub fn options(&self) -> &RemoteHubOptions {
        &self.options
    }

    /// Checks that the hub answers `2xx` on `GET {base}/health`.
    ///
    /// The body is ignored, so a hub may answer with `{}`, `"ok"` or nothing.
    ///
    /// # Errors
    ///
    /// The transport errors of the shared HTTP client, plus
    /// [`Error::validation`] for a `4xx` and [`Error::transport`] for any other
    /// non-2xx status.
    pub async fn health(&self) -> Result<()> {
        let response = self.send(HttpRequest::get(self.url(HEALTH_PATH))).await?;
        ensure_success(&response, HEALTH_ENDPOINT)
    }

    /// The absolute URL of `path` under the hub root.
    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.options.base_url)
    }

    /// Attaches the static headers and the bearer token to `request`.
    fn decorate(&self, mut request: HttpRequest) -> HttpRequest {
        request = request.header("accept", "application/json");
        for (name, value) in &self.options.headers {
            request = request.header(name.clone(), value.clone());
        }
        if let Some(api_key) = &self.options.api_key {
            request = request.bearer(SecretString::new(api_key.clone()));
        }
        request
    }

    /// Decorates and executes `request`, leaving the status to the caller.
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse> {
        self.client.execute(self.decorate(request)).await
    }

    /// Executes `GET {base}/v1/hub/bundles/{bundle_id}`.
    async fn bundle_request(&self, bundle_id: &str) -> Result<HttpResponse> {
        let path = format!("{BUNDLES_PATH}/{}", encode_segment(bundle_id));
        self.send(HttpRequest::get(self.url(&path))).await
    }
}

#[async_trait]
impl SharingBackend for RemoteHubSharingBackend {
    async fn publish(&self, bundle: &SharedSkillBundle) -> Result<String> {
        let request = HttpRequest::post(self.url(BUNDLES_PATH)).json(bundle)?;
        let response = self.send(request).await?;
        ensure_success(&response, PUBLISH_ENDPOINT)?;
        let reply: BundleIdReply = decode(&response, PUBLISH_ENDPOINT)?;
        let bundle_id = reply.bundle_id.trim();
        Ok(if bundle_id.is_empty() {
            bundle.bundle_id.clone()
        } else {
            bundle_id.to_owned()
        })
    }

    async fn list(&self) -> Result<Vec<SharedSkillBundle>> {
        let response = self.send(HttpRequest::get(self.url(BUNDLES_PATH))).await?;
        ensure_success(&response, LIST_ENDPOINT)?;
        let reply: BundleListReply = decode(&response, LIST_ENDPOINT)?;
        let mut bundles = reply.bundles;
        bundles.sort_by(|left, right| left.bundle_id.cmp(&right.bundle_id));
        Ok(bundles)
    }

    async fn get(&self, bundle_id: &str) -> Result<Option<SharedSkillBundle>> {
        let response = self.bundle_request(bundle_id).await?;
        if response.status == 404 {
            return Ok(None);
        }
        ensure_success(&response, GET_ENDPOINT)?;
        Ok(Some(decode(&response, GET_ENDPOINT)?))
    }

    async fn remove(&self, bundle_id: &str) -> Result<bool> {
        let path = format!("{BUNDLES_PATH}/{}", encode_segment(bundle_id));
        let response = self.send(HttpRequest::delete(self.url(&path))).await?;
        if response.status == 404 {
            return Ok(false);
        }
        ensure_success(&response, REMOVE_ENDPOINT)?;
        let reply: RemovedReply = decode(&response, REMOVE_ENDPOINT)?;
        Ok(reply.removed)
    }

    async fn search(&self, query: &QueryKeywords, limit: usize) -> Result<Vec<SkillSearchResult>> {
        if query.is_empty() {
            return Ok(Vec::new());
        }
        let body = SearchRequest { query, limit };
        let response = self
            .send(HttpRequest::post(self.url(SEARCH_PATH)).json(&body)?)
            .await?;
        ensure_success(&response, SEARCH_ENDPOINT)?;
        let reply: SearchReply = decode(&response, SEARCH_ENDPOINT)?;
        Ok(rank_results(reply.results, limit))
    }
}

/// The reply of `POST /v1/hub/bundles`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
struct BundleIdReply {
    /// The id the hub stored the bundle under; a blank value falls back to the
    /// bundle's content-derived id.
    #[serde(default)]
    bundle_id: String,
}

/// The reply of `GET /v1/hub/bundles`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
struct BundleListReply {
    /// The stored bundles, in whatever order the hub chose.
    #[serde(default)]
    bundles: Vec<SharedSkillBundle>,
}

/// The reply of `DELETE /v1/hub/bundles/{bundle_id}`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
struct RemovedReply {
    /// Whether the bundle existed.
    #[serde(default)]
    removed: bool,
}

/// The body of `POST /v1/hub/bundles/search`.
#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
struct SearchRequest<'a> {
    /// The query keywords.
    query: &'a QueryKeywords,
    /// The requested result count; `0` means unlimited.
    limit: usize,
}

/// The reply of `POST /v1/hub/bundles/search`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
struct SearchReply {
    /// The ranked rows, in whatever order and count the hub chose.
    #[serde(default)]
    results: Vec<SkillSearchResult>,
}

/// The error document a hub may answer a failed request with.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
struct ErrorEnvelope {
    /// The hub's error detail.
    error: HubError,
}

/// The hub error detail.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
struct HubError {
    /// The machine-readable error code.
    #[serde(default)]
    code: String,
    /// The human-readable error message.
    #[serde(default)]
    message: String,
}

/// Trims `raw`, requires an `http`/`https` scheme and a non-empty host, and
/// removes trailing `/` characters.
fn normalise_base_url(raw: &str) -> Result<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(Error::validation("sharing hub base_url must not be empty"));
    }
    let Some(rest) = trimmed
        .strip_prefix("https://")
        .or_else(|| trimmed.strip_prefix("http://"))
    else {
        let scheme = trimmed.split("://").next().unwrap_or_default();
        return Err(Error::validation(format!(
            "sharing hub base_url {trimmed:?} must use the http or https scheme, got {scheme:?}"
        )));
    };
    let authority_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let authority = &rest[..authority_end];
    let host_port = authority.rsplit('@').next().unwrap_or(authority);
    let host = host_port.split(':').next().unwrap_or_default();
    if host.is_empty() || host.chars().any(char::is_whitespace) {
        return Err(Error::validation(format!(
            "sharing hub base_url {trimmed:?} has no host"
        )));
    }
    Ok(trimmed.trim_end_matches('/').to_owned())
}

/// Percent-encodes one path segment, keeping the unreserved characters.
fn encode_segment(segment: &str) -> String {
    /// The uppercase hexadecimal digits.
    const HEX: &[u8; 16] = b"0123456789ABCDEF";

    let mut encoded = String::with_capacity(segment.len());
    for byte in segment.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                encoded.push(char::from(byte));
            }
            _ => {
                encoded.push('%');
                encoded.push(char::from(HEX[usize::from(byte >> 4)]));
                encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
            }
        }
    }
    encoded
}

/// Turns a non-2xx `response` into the documented error.
fn ensure_success(response: &HttpResponse, endpoint: &str) -> Result<()> {
    if response.is_success() {
        return Ok(());
    }
    Err(failure(response, endpoint))
}

/// The error for a non-2xx `response`: validation for `4xx`, transport otherwise.
fn failure(response: &HttpResponse, endpoint: &str) -> Error {
    let status = response.status;
    let detail = error_detail(response);
    let message = format!("sharing hub {endpoint} failed with HTTP {status}: {detail}");
    if (400..500).contains(&status) {
        Error::validation(message)
    } else {
        Error::transport(message)
    }
}

/// The hub's error detail, from its envelope when the body carries one.
fn error_detail(response: &HttpResponse) -> String {
    if let Ok(envelope) = response.json::<ErrorEnvelope>() {
        let code = envelope.error.code.trim();
        let message = envelope.error.message.trim();
        match (code.is_empty(), message.is_empty()) {
            (false, false) => return format!("{code}: {message}"),
            (false, true) => return code.to_owned(),
            (true, false) => return message.to_owned(),
            (true, true) => {}
        }
    }
    let body = response.text();
    let flat = body.split_whitespace().collect::<Vec<&str>>().join(" ");
    if flat.is_empty() {
        return "no error body".to_owned();
    }
    let mut snippet: String = flat.chars().take(ERROR_BODY_CHARS).collect();
    if flat.chars().count() > ERROR_BODY_CHARS {
        snippet.push_str("...");
    }
    snippet
}

/// Decodes a success body, naming `endpoint` when the payload is malformed.
fn decode<T: DeserializeOwned>(response: &HttpResponse, endpoint: &str) -> Result<T> {
    response.json::<T>().map_err(|error| {
        Error::validation(format!(
            "sharing hub {endpoint} returned a malformed body: {}",
            error.message()
        ))
    })
}

/// Filters, ranks and truncates hub search rows to the local backend's rules.
fn rank_results(mut results: Vec<SkillSearchResult>, limit: usize) -> Vec<SkillSearchResult> {
    results.retain(|result| result.score > 0.0);
    results.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.bundle_id.cmp(&right.bundle_id))
    });
    if limit > 0 {
        results.truncate(limit);
    }
    results
}

#[cfg(test)]
mod tests;
