//! The provider-agnostic HTTP client, request builder, and response types.
//!
//! [`HttpClient`] wraps a [`reqwest::Client`] and adds the retry/backoff policy
//! required by the LLM layers. Requests are built with [`HttpRequest`], which
//! supports query parameters, headers, JSON/text/byte bodies, a bearer token, a
//! per-request timeout, and an optional [`CancellationToken`].
//!
//! Non-2xx responses are returned as ordinary [`HttpResponse`] values (Python
//! parity: `raise_for_status=False` by default); call
//! [`HttpResponse::error_for_status`] to turn them into errors.

use std::collections::BTreeMap;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use bytes::{Bytes, BytesMut};
use futures::stream::{BoxStream, StreamExt};
use futures::Stream;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, CONTENT_TYPE};
use serde::de::DeserializeOwned;
use serde::Serialize;
use tokio_util::sync::CancellationToken;

use crate::config::env;
use crate::error::{Error, Result};
use crate::http::config::HttpClientConfig;
use crate::security::redact::sanitize_headers;
use crate::security::{redact_url_password, SecretString};

/// The maximum number of redirects followed when enabled.
const MAX_REDIRECTS: usize = 10;

/// A request body in one of the shapes the LLM layers need.
#[derive(Debug, Clone, PartialEq)]
pub enum HttpBody {
    /// A structured JSON document, serialized on send.
    Json(serde_json::Value),
    /// A plain-text body.
    Text(String),
    /// Raw bytes.
    Bytes(Bytes),
}

/// A fully described HTTP request, built with chained setters.
///
/// ```
/// use openjiuwen_common::http::HttpRequest;
///
/// let request = HttpRequest::post("https://api.example.com/v1/chat")
///     .query("stream", "false")
///     .header("X-Trace", "abc")
///     .json(&serde_json::json!({"model": "demo"}))
///     .unwrap();
///
/// assert_eq!(request.url(), "https://api.example.com/v1/chat");
/// ```
#[derive(Debug, Clone)]
pub struct HttpRequest {
    method: reqwest::Method,
    url: String,
    query: Vec<(String, String)>,
    headers: BTreeMap<String, String>,
    body: Option<HttpBody>,
    timeout: Option<Duration>,
    bearer_token: Option<SecretString>,
    cancellation: Option<CancellationToken>,
}

impl HttpRequest {
    /// Builds a request with the given method and URL.
    pub fn new(method: reqwest::Method, url: impl Into<String>) -> Self {
        Self {
            method,
            url: url.into(),
            query: Vec::new(),
            headers: BTreeMap::new(),
            body: None,
            timeout: None,
            bearer_token: None,
            cancellation: None,
        }
    }

    /// Builds a `GET` request.
    pub fn get(url: impl Into<String>) -> Self {
        Self::new(reqwest::Method::GET, url)
    }

    /// Builds a `POST` request.
    pub fn post(url: impl Into<String>) -> Self {
        Self::new(reqwest::Method::POST, url)
    }

    /// Builds a `PUT` request.
    pub fn put(url: impl Into<String>) -> Self {
        Self::new(reqwest::Method::PUT, url)
    }

    /// Builds a `PATCH` request.
    pub fn patch(url: impl Into<String>) -> Self {
        Self::new(reqwest::Method::PATCH, url)
    }

    /// Builds a `DELETE` request.
    pub fn delete(url: impl Into<String>) -> Self {
        Self::new(reqwest::Method::DELETE, url)
    }

    /// Builds a `HEAD` request.
    pub fn head(url: impl Into<String>) -> Self {
        Self::new(reqwest::Method::HEAD, url)
    }

    /// Appends one query parameter.
    #[must_use]
    pub fn query(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.query.push((key.into(), value.into()));
        self
    }

    /// Appends several query parameters.
    #[must_use]
    pub fn queries(mut self, params: impl IntoIterator<Item = (String, String)>) -> Self {
        self.query.extend(params);
        self
    }

    /// Appends one header.
    #[must_use]
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }

    /// Appends several headers.
    ///
    /// Named `with_headers` rather than `headers` because the read accessor
    /// [`HttpRequest::headers`] already owns that name (Rust does not allow two
    /// inherent methods with the same name).
    #[must_use]
    pub fn with_headers(mut self, headers: impl IntoIterator<Item = (String, String)>) -> Self {
        self.headers.extend(headers);
        self
    }

    /// Sets a JSON body.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the value cannot be serialized.
    pub fn json(mut self, body: &impl Serialize) -> Result<Self> {
        let value = serde_json::to_value(body)
            .map_err(|err| Error::validation(format!("serializing JSON request body: {err}")))?;
        self.body = Some(HttpBody::Json(value));
        Ok(self)
    }

    /// Sets a plain-text body.
    #[must_use]
    pub fn text(mut self, body: impl Into<String>) -> Self {
        self.body = Some(HttpBody::Text(body.into()));
        self
    }

    /// Sets a raw-bytes body.
    #[must_use]
    pub fn bytes(mut self, body: Bytes) -> Self {
        self.body = Some(HttpBody::Bytes(body));
        self
    }

    /// Sets the bearer token used for the `Authorization` header.
    #[must_use]
    pub fn bearer(mut self, token: SecretString) -> Self {
        self.bearer_token = Some(token);
        self
    }

    /// Overrides the client timeout for this request.
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Attaches a cancellation token to this request.
    #[must_use]
    pub fn cancel(mut self, cancellation: CancellationToken) -> Self {
        self.cancellation = Some(cancellation);
        self
    }

    /// Returns the request method.
    #[must_use]
    pub fn method(&self) -> &reqwest::Method {
        &self.method
    }

    /// Returns the request URL.
    #[must_use]
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Returns the request headers.
    #[must_use]
    pub fn headers(&self) -> &BTreeMap<String, String> {
        &self.headers
    }

    /// Returns the request body, if any.
    #[must_use]
    pub fn body(&self) -> Option<&HttpBody> {
        self.body.as_ref()
    }

    /// Returns the request cancellation token, if any.
    #[must_use]
    pub fn cancellation(&self) -> Option<&CancellationToken> {
        self.cancellation.as_ref()
    }
}

/// A streaming response body.
pub type HttpStream = BoxStream<'static, Result<Bytes>>;

/// A buffered HTTP response.
#[derive(Debug, Clone)]
pub struct HttpResponse {
    /// The numeric HTTP status code.
    pub status: u16,
    /// The response headers.
    pub headers: HeaderMap,
    /// The final URL after redirects.
    pub url: String,
    /// The response body.
    pub body: Bytes,
}

impl HttpResponse {
    /// Returns `true` for a 2xx status.
    #[must_use]
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }

    /// Decodes the body as UTF-8 (lossy) text.
    #[must_use]
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }

    /// Deserializes the body as JSON.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the body is not valid JSON for `T`.
    pub fn json<T: DeserializeOwned>(&self) -> Result<T> {
        serde_json::from_slice(&self.body).map_err(|err| {
            Error::validation(format!(
                "decoding JSON response from {}: {err}",
                redact_credentials(&self.url)
            ))
        })
    }

    /// Returns the raw response body.
    #[must_use]
    pub fn bytes(&self) -> &Bytes {
        &self.body
    }

    /// Returns `self` for a 2xx status, otherwise a transport error.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Transport`] listing the status code and URL for any
    /// non-2xx response.
    pub fn error_for_status(self) -> Result<Self> {
        if self.is_success() {
            Ok(self)
        } else {
            Err(Error::transport(format!(
                "HTTP {} for {}",
                self.status,
                redact_credentials(&self.url)
            )))
        }
    }
}

/// A response whose body has not been buffered yet.
pub struct HttpStreamResponse {
    /// The numeric HTTP status code.
    pub status: u16,
    /// The response headers.
    pub headers: HeaderMap,
    /// The final URL after redirects.
    pub url: String,
    /// The streaming body; dropping it cancels the transfer.
    pub body: HttpStream,
}

/// A provider-agnostic asynchronous HTTP client with retry support.
#[derive(Debug, Clone)]
pub struct HttpClient {
    inner: reqwest::Client,
    config: HttpClientConfig,
}

impl HttpClient {
    /// Builds a client from `config`.
    ///
    /// Default headers are sanitized with
    /// [`crate::security::sanitize_headers`] (so a caller-supplied
    /// `Authorization` header is dropped), the proxy falls back to
    /// `http_proxy`/`https_proxy`, and the `NO_PROXY` bypass list is honoured.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for invalid header names/values or an
    /// invalid proxy URL, and [`Error::Config`] when the client cannot be built.
    pub fn new(config: HttpClientConfig) -> Result<Self> {
        let mut builder = reqwest::Client::builder()
            .timeout(config.timeout)
            .danger_accept_invalid_certs(!config.verify_ssl)
            .redirect(if config.follow_redirects {
                reqwest::redirect::Policy::limited(MAX_REDIRECTS)
            } else {
                reqwest::redirect::Policy::none()
            });

        if let Some(connect_timeout) = config.connect_timeout {
            builder = builder.connect_timeout(connect_timeout);
        }
        if let Some(user_agent) = &config.user_agent {
            builder = builder.user_agent(user_agent);
        }

        let sanitized = sanitize_headers(&config.headers);
        if !sanitized.is_empty() {
            let mut headers = HeaderMap::new();
            for (name, value) in &sanitized {
                headers.insert(parse_header_name(name)?, parse_header_value(name, value)?);
            }
            builder = builder.default_headers(headers);
        }

        let proxy_url = config.proxy.clone().or_else(env::http_proxy);
        if let Some(proxy_url) = proxy_url {
            let proxy = reqwest::Proxy::all(&proxy_url).map_err(|err| {
                Error::validation(format!(
                    "invalid proxy URL {:?}: {err}",
                    redact_credentials(&proxy_url)
                ))
            })?;
            let no_proxy = env::no_proxy()
                .as_deref()
                .and_then(reqwest::NoProxy::from_string);
            builder = builder.proxy(proxy.no_proxy(no_proxy));
        }

        let inner = builder
            .build()
            .map_err(|err| Error::config(format!("building HTTP client: {err}")))?;
        Ok(Self { inner, config })
    }

    /// Builds a client with [`HttpClientConfig::default`].
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`HttpClient::new`].
    pub fn with_defaults() -> Result<Self> {
        Self::new(HttpClientConfig::default())
    }

    /// Returns the configuration this client was built with.
    #[must_use]
    pub fn config(&self) -> &HttpClientConfig {
        &self.config
    }

    /// Returns the underlying [`reqwest::Client`].
    #[must_use]
    pub fn inner(&self) -> &reqwest::Client {
        &self.inner
    }

    /// Starts building a request for `method` and `url`.
    #[must_use]
    pub fn request(&self, method: reqwest::Method, url: impl Into<String>) -> HttpRequest {
        HttpRequest::new(method, url)
    }

    /// Executes a request and buffers the response body.
    ///
    /// Retries are driven by [`HttpClientConfig::retry_policy`]. Non-2xx responses
    /// are returned as ordinary [`HttpResponse`] values.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid URL or header, [`Error::Timeout`]
    /// when the deadline expires, [`Error::Transport`] for connection/body failures
    /// or an oversized response, and [`Error::Cancelled`] when the request token is
    /// cancelled.
    pub async fn execute(&self, request: HttpRequest) -> Result<HttpResponse> {
        let response = self.send_with_retry(&request).await?;
        self.read_response(response).await
    }

    /// Executes a request, racing it against an external cancellation token.
    ///
    /// A token carried by `request` is also honoured.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`HttpClient::execute`], plus [`Error::Cancelled`]
    /// when either token is cancelled.
    pub async fn execute_cancellable(
        &self,
        request: HttpRequest,
        cancel: CancellationToken,
    ) -> Result<HttpResponse> {
        tokio::select! {
            biased;
            () = cancel.cancelled() => Err(Error::cancelled()),
            result = self.execute(request) => result,
        }
    }

    /// Executes a `GET` request.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`HttpClient::execute`].
    pub async fn get(&self, url: impl Into<String>) -> Result<HttpResponse> {
        self.execute(HttpRequest::get(url)).await
    }

    /// Executes a `GET` request and deserializes the JSON body.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`HttpClient::execute`] and [`HttpResponse::json`].
    pub async fn get_json<T: DeserializeOwned>(&self, url: impl Into<String>) -> Result<T> {
        self.get(url).await?.json()
    }

    /// Executes a `POST` request with a JSON body.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`HttpRequest::json`] and [`HttpClient::execute`].
    pub async fn post_json<B: Serialize + Sync>(
        &self,
        url: impl Into<String>,
        body: &B,
    ) -> Result<HttpResponse> {
        self.execute(HttpRequest::post(url).json(body)?).await
    }

    /// Executes a request and returns the still-streaming response.
    ///
    /// Retries happen before the body is handed out. Dropping the returned stream
    /// cancels the transfer; a token carried by `request` turns into
    /// [`Error::Cancelled`] once it fires.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`HttpClient::execute`] before the body starts.
    pub async fn stream(&self, request: HttpRequest) -> Result<HttpStreamResponse> {
        let cancellation = request.cancellation.clone();
        let response = self.send_with_retry(&request).await?;
        Ok(stream_response(response, cancellation))
    }

    /// Validates the URL, builds the request, and runs the retry loop.
    async fn send_with_retry(&self, request: &HttpRequest) -> Result<reqwest::Response> {
        let built = self.build_request(request)?;
        let cancellation = request.cancellation.clone();
        let policy = self.config.retry_policy();
        let mut attempt = 0_u32;

        loop {
            let attempt_request = built.try_clone().ok_or_else(|| {
                Error::transport("request body cannot be replayed for a retry attempt")
            })?;
            match self.send(attempt_request, cancellation.as_ref()).await {
                Ok(response) => {
                    let status = response.status().as_u16();
                    if policy.should_retry(attempt, None, Some(status)) {
                        drop(response);
                        tokio::time::sleep(policy.backoff(attempt)).await;
                        attempt += 1;
                        continue;
                    }
                    return Ok(response);
                }
                Err(error) => {
                    if policy.should_retry(attempt, Some(&error), None) {
                        tokio::time::sleep(policy.backoff(attempt)).await;
                        attempt += 1;
                        continue;
                    }
                    return Err(error);
                }
            }
        }
    }

    /// Sends one attempt, racing the token when present.
    async fn send(
        &self,
        request: reqwest::Request,
        cancellation: Option<&CancellationToken>,
    ) -> Result<reqwest::Response> {
        let send = self.inner.execute(request);
        match cancellation {
            Some(token) => {
                tokio::select! {
                    biased;
                    () = token.cancelled() => Err(Error::cancelled()),
                    result = send => result.map_err(Error::from),
                }
            }
            None => send.await.map_err(Error::from),
        }
    }

    /// Translates an [`HttpRequest`] into a replayable [`reqwest::Request`].
    fn build_request(&self, request: &HttpRequest) -> Result<reqwest::Request> {
        let url = validate_url(&request.url)?;
        let mut builder = self
            .inner
            .request(request.method.clone(), url)
            .query(&request.query)
            .timeout(request.timeout.unwrap_or(self.config.timeout));

        for (name, value) in &request.headers {
            builder = builder.header(parse_header_name(name)?, parse_header_value(name, value)?);
        }
        if let Some(token) = &request.bearer_token {
            builder = builder.bearer_auth(token.expose_secret());
        }

        if let Some(body) = &request.body {
            builder = match body {
                HttpBody::Json(value) => {
                    let bytes = serde_json::to_vec(value).map_err(|err| {
                        Error::validation(format!("serializing JSON request body: {err}"))
                    })?;
                    if has_header(&request.headers, CONTENT_TYPE.as_str()) {
                        builder
                    } else {
                        builder.header(CONTENT_TYPE, "application/json")
                    }
                    .body(bytes)
                }
                HttpBody::Text(text) => builder.body(text.clone()),
                HttpBody::Bytes(bytes) => builder.body(bytes.clone()),
            };
        }

        builder.build().map_err(Error::from)
    }

    /// Buffers a response body, enforcing `max_response_bytes`.
    async fn read_response(&self, response: reqwest::Response) -> Result<HttpResponse> {
        let status = response.status().as_u16();
        let headers = response.headers().clone();
        let url = response.url().to_string();
        let body = self.read_body(response).await?;
        Ok(HttpResponse {
            status,
            headers,
            url,
            body,
        })
    }

    /// Reads a response body, checking the declared and actual lengths.
    async fn read_body(&self, response: reqwest::Response) -> Result<Bytes> {
        let limit = self.config.max_response_bytes;
        if let Some(limit) = limit {
            if let Some(length) = response.content_length() {
                if length > limit {
                    return Err(response_too_large(limit));
                }
            }
        }

        let mut stream = response.bytes_stream();
        let mut buffer = BytesMut::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(Error::from)?;
            buffer.extend_from_slice(&chunk);
            if let Some(limit) = limit {
                if u64::try_from(buffer.len()).unwrap_or(u64::MAX) > limit {
                    return Err(response_too_large(limit));
                }
            }
        }
        Ok(buffer.freeze())
    }
}

/// Builds a streaming response from a raw response.
fn stream_response(
    response: reqwest::Response,
    cancellation: Option<CancellationToken>,
) -> HttpStreamResponse {
    let status = response.status().as_u16();
    let headers = response.headers().clone();
    let url = response.url().to_string();
    let body: HttpStream = Box::pin(
        response
            .bytes_stream()
            .map(|chunk| chunk.map_err(Error::from)),
    );

    let body: HttpStream = match cancellation {
        Some(token) => Box::pin(CancellableStream::new(body, token)),
        None => body,
    };

    HttpStreamResponse {
        status,
        headers,
        url,
        body,
    }
}

/// A stream wrapper that yields [`Error::Cancelled`] once the token fires.
struct CancellableStream {
    inner: HttpStream,
    token: CancellationToken,
    finished: bool,
}

impl CancellableStream {
    fn new(inner: HttpStream, token: CancellationToken) -> Self {
        Self {
            inner,
            token,
            finished: false,
        }
    }
}

impl Stream for CancellableStream {
    type Item = Result<Bytes>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.finished {
            return Poll::Ready(None);
        }
        if this.token.is_cancelled() {
            this.finished = true;
            return Poll::Ready(Some(Err(Error::cancelled())));
        }

        let polled = this.inner.as_mut().poll_next(cx);
        if matches!(polled, Poll::Ready(None)) {
            this.finished = true;
        }
        polled
    }
}

/// Parses a header name, mapping failures to [`Error::Validation`].
fn parse_header_name(name: &str) -> Result<HeaderName> {
    HeaderName::from_bytes(name.as_bytes())
        .map_err(|err| Error::validation(format!("invalid HTTP header name {name:?}: {err}")))
}

/// Parses a header value, mapping failures to [`Error::Validation`].
fn parse_header_value(name: &str, value: &str) -> Result<HeaderValue> {
    HeaderValue::from_str(value)
        .map_err(|err| Error::validation(format!("invalid value for HTTP header {name:?}: {err}")))
}

/// Returns `true` when `headers` already contains `name` (case-insensitively).
fn has_header(headers: &BTreeMap<String, String>, name: &str) -> bool {
    headers.keys().any(|key| key.eq_ignore_ascii_case(name))
}

/// Validates the scheme and strips trailing whitespace from a URL.
fn validate_url(raw: &str) -> Result<String> {
    let trimmed = raw.trim_end();
    if trimmed.is_empty() {
        return Err(Error::validation("URL must not be empty"));
    }

    let parsed = url::Url::parse(trimmed).map_err(|err| {
        Error::validation(redact_credentials(&format!(
            "invalid URL {trimmed:?}: {err}"
        )))
    })?;
    match parsed.scheme() {
        "http" | "https" => Ok(trimmed.to_owned()),
        scheme => Err(Error::validation(format!(
            "unsupported URL scheme {scheme:?}: only http and https are allowed"
        ))),
    }
}

/// Masks every password embedded in a URL (or inside a message quoting one).
///
/// [`redact_url_password`] is the canonical helper, but it deliberately returns an
/// unparseable URL unchanged — and a malformed proxy URL is exactly the value that
/// reaches an error message. When the parsed helper leaves the text untouched, the
/// userinfo password is masked by scanning the authority of each `scheme://...`
/// occurrence, so `http://user:secret@proxy:port` (invalid port, hence unparseable)
/// is reported as `http://user:***@proxy:port`.
#[must_use]
fn redact_credentials(text: &str) -> String {
    let redacted = redact_url_password(text);
    if redacted != text {
        return redacted;
    }
    mask_userinfo_passwords(text).unwrap_or_else(|| text.to_owned())
}

/// Replaces the password of every `scheme://user:password@host` occurrence.
///
/// Returns `None` when the text contains no password to mask. Authorities end at
/// the characters that terminate a URL inside prose (`/ ? #`, quotes, brackets,
/// parentheses, whitespace, ...) so a message quoting two URLs masks both instead
/// of treating everything up to the last `@` as one authority.
#[must_use]
fn mask_userinfo_passwords(text: &str) -> Option<String> {
    let mut masked = String::with_capacity(text.len());
    let mut changed = false;
    let mut rest = text;

    while let Some(scheme) = rest.find("://") {
        let start = scheme + 3;
        let end = authority_end(rest, start);
        let authority = &rest[start..end];

        if let Some(at) = authority.rfind('@') {
            let userinfo = &authority[..at];
            if let Some((user, password)) = userinfo.split_once(':') {
                if !password.is_empty() {
                    masked.push_str(&rest[..start]);
                    masked.push_str(user);
                    masked.push_str(":***@");
                    masked.push_str(&authority[at + 1..]);
                    changed = true;
                    rest = &rest[end..];
                    continue;
                }
            }
        }

        // Nothing to mask in this authority; keep it verbatim and keep scanning.
        masked.push_str(&rest[..end]);
        rest = &rest[end..];
    }

    masked.push_str(rest);
    changed.then_some(masked)
}

/// Returns the offset where the URL authority starting at `start` ends.
fn authority_end(text: &str, start: usize) -> usize {
    text[start..]
        .find(|character: char| {
            character.is_whitespace() || "/?#\"'`<>(),;\\|}{^".contains(character)
        })
        .map_or(text.len(), |offset| start + offset)
}

/// The error returned when a body exceeds `max_response_bytes`.
fn response_too_large(limit: u64) -> Error {
    Error::transport(format!("response exceeds {limit} bytes"))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn client() -> HttpClient {
        HttpClient::with_defaults().unwrap()
    }

    #[test]
    fn request_builders_expose_their_fields() {
        let token = CancellationToken::new();
        let request = HttpRequest::post("https://example.com/v1")
            .query("a", "1")
            .queries(vec![("b".to_owned(), "2".to_owned())])
            .header("X-Test", "yes")
            .with_headers(vec![("X-Other".to_owned(), "no".to_owned())])
            .json(&serde_json::json!({"k": "v"}))
            .unwrap()
            .bearer(SecretString::from("sk-test"))
            .timeout(Duration::from_secs(3))
            .cancel(token.clone());

        assert_eq!(request.method().clone(), reqwest::Method::POST);
        assert_eq!(request.url(), "https://example.com/v1");
        assert_eq!(request.query.len(), 2);
        assert_eq!(
            request.headers().get("X-Test").map(String::as_str),
            Some("yes")
        );
        assert_eq!(
            request.headers().get("X-Other").map(String::as_str),
            Some("no")
        );
        assert_eq!(
            request.body(),
            Some(&HttpBody::Json(serde_json::json!({"k": "v"})))
        );
        assert_eq!(
            request.cancellation().map(CancellationToken::is_cancelled),
            Some(false)
        );
        assert!(!format!("{request:?}").contains("sk-test"));
    }

    #[test]
    fn text_and_bytes_bodies_are_recorded() {
        let text = HttpRequest::put("https://example.com").text("hello");
        assert_eq!(text.body(), Some(&HttpBody::Text("hello".to_owned())));

        let bytes = HttpRequest::patch("https://example.com").bytes(Bytes::from_static(b"raw"));
        assert_eq!(
            bytes.body(),
            Some(&HttpBody::Bytes(Bytes::from_static(b"raw")))
        );
    }

    #[test]
    fn error_for_status_success_and_failure() {
        let ok = HttpResponse {
            status: 204,
            headers: HeaderMap::new(),
            url: "https://example.com".to_owned(),
            body: Bytes::new(),
        };
        assert!(ok.is_success());
        let ok = ok.error_for_status().unwrap();
        assert_eq!(ok.status, 204);

        let failed = HttpResponse {
            status: 404,
            headers: HeaderMap::new(),
            url: "https://example.com/missing".to_owned(),
            body: Bytes::new(),
        };
        assert!(!failed.is_success());
        let error = failed.error_for_status().unwrap_err();
        assert_eq!(error.kind(), crate::error::ErrorKind::Transport);
        assert_eq!(
            error.to_string(),
            "transport: HTTP 404 for https://example.com/missing"
        );
    }

    #[test]
    fn response_json_decodes_and_reports_invalid_payloads() {
        let response = HttpResponse {
            status: 200,
            headers: HeaderMap::new(),
            url: "https://example.com".to_owned(),
            body: Bytes::from_static(br#"{"n": 3}"#),
        };
        let value: serde_json::Value = response.json().unwrap();
        assert_eq!(value["n"], 3);

        let invalid = HttpResponse {
            status: 200,
            headers: HeaderMap::new(),
            url: "https://example.com".to_owned(),
            body: Bytes::from_static(b"not json"),
        };
        let error = invalid.json::<serde_json::Value>().unwrap_err();
        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
    }

    #[test]
    fn response_text_is_lossy_and_bytes_round_trip() {
        let response = HttpResponse {
            status: 200,
            headers: HeaderMap::new(),
            url: "https://example.com".to_owned(),
            body: Bytes::from_static(b"hello"),
        };
        assert_eq!(response.text(), "hello");
        assert_eq!(response.bytes(), &Bytes::from_static(b"hello"));
    }

    #[test]
    fn validate_url_accepts_http_and_rejects_other_schemes() {
        assert_eq!(
            validate_url("https://example.com/path  ").unwrap(),
            "https://example.com/path"
        );
        assert_eq!(
            validate_url("http://example.com").unwrap(),
            "http://example.com"
        );

        assert_eq!(
            validate_url("ftp://example.com").unwrap_err().kind(),
            crate::error::ErrorKind::Validation
        );
        assert_eq!(
            validate_url("file:///tmp/x").unwrap_err().kind(),
            crate::error::ErrorKind::Validation
        );
        assert_eq!(
            validate_url("   ").unwrap_err().kind(),
            crate::error::ErrorKind::Validation
        );
    }

    #[tokio::test]
    async fn invalid_url_is_rejected_before_any_network_call() {
        let error = client()
            .execute(HttpRequest::get("ftp://example.com/file"))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
    }

    #[test]
    fn config_headers_drop_protected_entries() {
        let mut headers = BTreeMap::new();
        headers.insert("Authorization".to_owned(), "Bearer nope".to_owned());
        headers.insert("X-Custom".to_owned(), "value".to_owned());

        let sanitized = sanitize_headers(&headers);
        assert!(!sanitized.contains_key("Authorization"));
        assert_eq!(sanitized.get("X-Custom").map(String::as_str), Some("value"));
    }

    #[test]
    fn invalid_proxy_url_does_not_leak_its_password() {
        // `proxy:port` is not a numeric port, so the URL stays unparseable and the
        // parsed redaction helper cannot mask it.
        let config = HttpClientConfig::default().with_proxy("http://user:s3cr3t@proxy:port");
        let error = HttpClient::new(config).unwrap_err();

        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
        let display = error.to_string();
        let debug = format!("{error:?}");
        assert!(!display.contains("s3cr3t"), "display: {display}");
        assert!(!debug.contains("s3cr3t"), "debug: {debug}");
        assert!(
            display.contains("user:***@proxy:port"),
            "the password is masked in place: {display}"
        );
    }

    #[test]
    fn invalid_url_does_not_leak_its_password() {
        let error = validate_url("https://user:s3cr3t@example.com:badport/x").unwrap_err();
        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
        assert!(!error.to_string().contains("s3cr3t"), "{error}");
        assert!(
            error.to_string().contains("user:***@example.com:badport"),
            "{error}"
        );
    }

    #[test]
    fn status_and_json_errors_mask_credentials_in_the_url() {
        let failed = HttpResponse {
            status: 401,
            headers: HeaderMap::new(),
            url: "https://user:s3cr3t@example.com/missing".to_owned(),
            body: Bytes::new(),
        };
        let error = failed.error_for_status().unwrap_err();
        assert!(!error.to_string().contains("s3cr3t"), "{error}");
        assert_eq!(
            error.to_string(),
            "transport: HTTP 401 for https://user:***@example.com/missing"
        );

        let invalid = HttpResponse {
            status: 200,
            headers: HeaderMap::new(),
            url: "https://user:s3cr3t@example.com/v1".to_owned(),
            body: Bytes::from_static(b"not json"),
        };
        let error = invalid.json::<serde_json::Value>().unwrap_err();
        assert!(!error.to_string().contains("s3cr3t"), "{error}");
        assert!(
            error.to_string().contains("user:***@example.com"),
            "{error}"
        );
    }

    #[test]
    fn redact_credentials_masks_every_occurrence_and_leaves_clean_text_alone() {
        assert_eq!(
            redact_credentials("https://user:secret@example.com/a"),
            "https://user:***@example.com/a"
        );
        assert_eq!(
            redact_credentials("bad proxy \"http://u:p@h:port\" and echo (http://u:p@h:port) here"),
            "bad proxy \"http://u:***@h:port\" and echo (http://u:***@h:port) here"
        );
        // No credentials, no host:port to mask, or an empty password: unchanged.
        assert_eq!(
            redact_credentials("https://example.com/a?b=c#d"),
            "https://example.com/a?b=c#d"
        );
        assert_eq!(
            redact_credentials("max_retries must be >= 0"),
            "max_retries must be >= 0"
        );
        assert_eq!(
            redact_credentials("redis://:secret@host:6379/0"),
            "redis://:***@host:6379/0"
        );
        assert_eq!(redact_credentials(""), "");
    }
}
