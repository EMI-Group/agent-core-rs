//! Shared HTTP transport and error type for the remote sandbox providers.
//!
//! Python parity: every provider under
//! `openjiuwen/extensions/sys_operation/sandbox/providers/` builds its own HTTP
//! session, injects the card's `auth_headers` / `auth_query_params`, and maps
//! non-success responses onto an `OpResult` failure. Rust hoists that into one
//! type — [`SandboxHttp`] — so the `AIO`, `JiuwenBox`, and `YuanRong` providers
//! differ
//! only in their URL shapes and payload schemas.
//!
//! Failures are [`RemoteError`] values: [`gateway_failure`] renders one as the
//! kernel's `SandboxGatewayError` (code `199007`) result, while
//! [`remote_error`] turns one into an [`openjiuwen_common::error::Error`] for
//! providers that surface `Result`. Credentials and payload contents are never
//! logged; request failures are logged at `debug` without the URL, because the
//! URL carries the configured auth query parameters.

use std::collections::BTreeMap;
use std::time::Duration;

use base64::Engine as _;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_core::sys_operation::result::OpResult;
use openjiuwen_core::sys_operation::sandbox::config::{SandboxEndpoint, SandboxGatewayConfig};
use openjiuwen_core::sys_operation::status::{sandbox_gateway_message, SysOpStatus};

/// Log target shared by the remote sandbox providers.
const LOG_TARGET: &str = "openjiuwen_extensions::sys_operation";

/// Why a remote sandbox provider call failed.
#[derive(Debug, Clone)]
pub(crate) enum RemoteError {
    /// The sandbox container is gone (provider-specific 404 body) — callers may recreate.
    ContainerGone(String),
    /// The remote path does not exist.
    NotFound(String),
    /// The remote destination already exists.
    AlreadyExists(String),
    /// The provider answered with a non-success HTTP status.
    Status {
        /// The HTTP status code.
        status: u16,
        /// Message including the status and any provider detail.
        message: String,
    },
    /// No HTTP response (connection refused, timeout, body/JSON decode failure).
    Transport(String),
    /// The provider rejected the request because a search exceeded its result limit.
    SearchLimit(String),
    /// Invalid configuration or parameters detected before/while talking to the provider.
    Invalid(String),
}

impl RemoteError {
    /// Human-readable description, including the status where relevant.
    pub(crate) fn message(&self) -> String {
        match self {
            Self::ContainerGone(message)
            | Self::NotFound(message)
            | Self::AlreadyExists(message)
            | Self::Transport(message)
            | Self::SearchLimit(message)
            | Self::Invalid(message) => message.clone(),
            Self::Status { status, message } => {
                if message.starts_with("HTTP ") {
                    message.clone()
                } else {
                    format!("HTTP {status}: {message}")
                }
            }
        }
    }

    /// The HTTP status behind the failure, when one is known.
    pub(crate) fn status(&self) -> Option<u16> {
        match self {
            Self::NotFound(_) => Some(404),
            Self::Status { status, .. } => Some(*status),
            Self::ContainerGone(_)
            | Self::AlreadyExists(_)
            | Self::Transport(_)
            | Self::SearchLimit(_)
            | Self::Invalid(_) => None,
        }
    }

    /// Whether the provider reported the sandbox container as gone.
    pub(crate) fn is_container_gone(&self) -> bool {
        matches!(self, Self::ContainerGone(_))
    }

    /// Whether the remote path does not exist.
    pub(crate) fn is_not_found(&self) -> bool {
        matches!(self, Self::NotFound(_))
    }
}

/// An HTTP client bound to one sandbox endpoint.
///
/// The base URL, the card's auth headers, and the card's auth query parameters
/// are resolved once at construction; every request then goes through
/// [`SandboxHttp::request`].
pub(crate) struct SandboxHttp {
    client: reqwest::Client,
    base_url: String,
    auth_headers: BTreeMap<String, String>,
    auth_query: BTreeMap<String, String>,
    provider: &'static str,
    timeout: Duration,
}

impl SandboxHttp {
    /// Builds a client for `endpoint`, falling back to the launcher's base URL.
    ///
    /// `provider` is the sandbox type (`"aio"`, `"jiuwenbox"`, `"yuanrong"`) and
    /// appears in messages. A value without a scheme gets `http://`; trailing
    /// slashes are stripped.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Config`](openjiuwen_common::error::ErrorKind::Config)
    /// when neither the endpoint nor the launcher configuration carries a base
    /// URL, and when the HTTP client cannot be built.
    pub(crate) fn new(
        provider: &'static str,
        endpoint: &SandboxEndpoint,
        config: &SandboxGatewayConfig,
    ) -> Result<Self> {
        let mut base_url = endpoint.base_url.trim().to_string();
        if base_url.is_empty() {
            base_url = config
                .launcher_config
                .as_ref()
                .and_then(|launcher| launcher.base_url.as_deref())
                .unwrap_or_default()
                .trim()
                .to_string();
        }
        if base_url.is_empty() {
            return Err(Error::config(format!(
                "{provider} provider requires endpoint.base_url"
            )));
        }
        let base_url = normalize_base_url(&base_url);
        let timeout = Duration::from_secs(config.timeout_seconds.max(1));
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|error| {
                Error::config(format!(
                    "{provider} provider could not build the HTTP client: {error}"
                ))
            })?;
        Ok(Self {
            client,
            base_url,
            auth_headers: config.auth_headers.clone(),
            auth_query: config.auth_query_params.clone(),
            provider,
            timeout,
        })
    }

    /// The normalized base URL (no trailing slash, scheme included).
    pub(crate) fn base_url(&self) -> &str {
        &self.base_url
    }

    /// The sandbox type this client speaks to.
    pub(crate) fn provider(&self) -> &'static str {
        self.provider
    }

    /// The configured request timeout.
    pub(crate) fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Adds a default `Authorization: Bearer <token>` header.
    ///
    /// A configured `auth_headers` entry named `authorization`
    /// (case-insensitive) wins, and an empty or whitespace-only token is ignored.
    pub(crate) fn with_bearer_token(mut self, token: Option<String>) -> Self {
        let token = token.map(|token| token.trim().to_string());
        if let Some(token) = token.filter(|token| !token.is_empty()) {
            let configured = self
                .auth_headers
                .keys()
                .any(|key| key.eq_ignore_ascii_case("authorization"));
            if !configured {
                self.auth_headers
                    .insert("Authorization".to_string(), format!("Bearer {token}"));
            }
        }
        self
    }

    /// Joins `path` to the base URL and appends the auth query parameters.
    pub(crate) fn url(&self, path: &str) -> String {
        let mut url = format!("{}{}", self.base_url, path);
        if self.auth_query.is_empty() {
            return url;
        }
        url.push(if url.contains('?') { '&' } else { '?' });
        let query = self
            .auth_query
            .iter()
            .map(|(key, value)| {
                format!("{}={}", percent_encode(key), percent_encode(value.as_str()))
            })
            .collect::<Vec<_>>()
            .join("&");
        url.push_str(&query);
        url
    }

    /// A request builder for `method` + `path` with every auth header applied.
    pub(crate) fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let mut builder = self.client.request(method, self.url(path));
        for (key, value) in &self.auth_headers {
            builder = builder.header(key.as_str(), value.as_str());
        }
        builder
    }

    /// Sends the request and decodes the JSON body into `T`.
    ///
    /// # Errors
    ///
    /// Returns [`RemoteError::NotFound`] for 404, [`RemoteError::Status`] for any
    /// other non-2xx answer, and [`RemoteError::Transport`] when no usable
    /// response arrives.
    pub(crate) async fn send_json<T: serde::de::DeserializeOwned>(
        &self,
        request: reqwest::RequestBuilder,
        operation: &str,
    ) -> std::result::Result<T, RemoteError> {
        let body = self.send_success_body(request, operation).await?;
        serde_json::from_slice(&body).map_err(|error| {
            RemoteError::Transport(format!(
                "{operation}: could not decode the {} response as JSON: {error}",
                self.provider
            ))
        })
    }

    /// Sends the request and decodes the JSON body as a [`serde_json::Value`].
    pub(crate) async fn send_value(
        &self,
        request: reqwest::RequestBuilder,
        operation: &str,
    ) -> std::result::Result<serde_json::Value, RemoteError> {
        self.send_json(request, operation).await
    }

    /// Sends the request and returns the raw body bytes.
    pub(crate) async fn send_bytes(
        &self,
        request: reqwest::RequestBuilder,
        operation: &str,
    ) -> std::result::Result<Vec<u8>, RemoteError> {
        self.send_success_body(request, operation).await
    }

    /// Sends the request and discards the body.
    pub(crate) async fn send_empty(
        &self,
        request: reqwest::RequestBuilder,
        operation: &str,
    ) -> std::result::Result<(), RemoteError> {
        self.send_success_body(request, operation).await.map(|_| ())
    }

    /// Sends the request and returns the status and body without status mapping.
    ///
    /// Providers that own their status handling (`YuanRong` ADX, `JiuwenBox`'s
    /// empty-body 404s) use this and inspect [`RawResponse`] themselves.
    pub(crate) async fn send_raw(
        &self,
        request: reqwest::RequestBuilder,
        operation: &str,
    ) -> std::result::Result<RawResponse, RemoteError> {
        let (status, body) = self.execute(request, operation).await?;
        Ok(RawResponse { status, body })
    }

    /// Sends the request, mapping non-2xx answers to a [`RemoteError`].
    async fn send_success_body(
        &self,
        request: reqwest::RequestBuilder,
        operation: &str,
    ) -> std::result::Result<Vec<u8>, RemoteError> {
        let (status, body) = self.execute(request, operation).await?;
        if is_success(status) {
            Ok(body)
        } else {
            Err(status_error(status, &body, operation))
        }
    }

    /// Sends the request and collects `(status, body)`.
    async fn execute(
        &self,
        request: reqwest::RequestBuilder,
        operation: &str,
    ) -> std::result::Result<(u16, Vec<u8>), RemoteError> {
        let response = request.send().await.map_err(|error| {
            // The reqwest error carries the request URL, which includes the auth
            // query parameters: strip it before it reaches a log or a message.
            let error = error.without_url();
            tracing::debug!(
                target: LOG_TARGET,
                provider = self.provider,
                operation,
                "remote sandbox request failed"
            );
            RemoteError::Transport(format!(
                "{operation}: {} request failed: {error}",
                self.provider
            ))
        })?;
        let status = response.status().as_u16();
        let body = response.bytes().await.map_err(|error| {
            let error = error.without_url();
            tracing::debug!(
                target: LOG_TARGET,
                provider = self.provider,
                operation,
                "remote sandbox response body could not be read"
            );
            RemoteError::Transport(format!(
                "{operation}: {} response body could not be read: {error}",
                self.provider
            ))
        })?;
        Ok((status, body.to_vec()))
    }
}

/// A response with its status and body, left unmapped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RawResponse {
    /// The HTTP status code.
    pub status: u16,
    /// The raw response body.
    pub body: Vec<u8>,
}

impl RawResponse {
    /// Parses the body as JSON, or `None` when it is empty or not JSON.
    pub(crate) fn json(&self) -> Option<serde_json::Value> {
        if self.body.is_empty() {
            return None;
        }
        serde_json::from_slice(&self.body).ok()
    }

    /// The body as lossy UTF-8 text.
    pub(crate) fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).to_string()
    }

    /// Whether the status is 2xx.
    pub(crate) fn is_success(&self) -> bool {
        is_success(self.status)
    }
}

/// Maps a [`RemoteError`] onto the kernel's `SandboxGatewayError` (code `199007`).
pub(crate) fn gateway_failure<T>(operation: &str, error: &RemoteError) -> OpResult<T> {
    OpResult::failure(
        SysOpStatus::SandboxGatewayError,
        sandbox_gateway_message(operation, &error.message()),
    )
}

/// Maps a [`RemoteError`] onto a plain [`openjiuwen_common::error::Error`].
pub(crate) fn remote_error(
    operation: &str,
    error: &RemoteError,
) -> openjiuwen_common::error::Error {
    match error {
        RemoteError::NotFound(message) => Error::not_found(format!("{operation}: {message}")),
        RemoteError::Invalid(message) => Error::config(format!("{operation}: {message}")),
        RemoteError::ContainerGone(_)
        | RemoteError::AlreadyExists(_)
        | RemoteError::Status { .. }
        | RemoteError::Transport(_)
        | RemoteError::SearchLimit(_) => Error::sys_op(format!("{operation}: {}", error.message())),
    }
}

/// Lowercase hex encoding (no `hex` crate in the workspace).
pub(crate) fn hex_encode(bytes: &[u8]) -> String {
    const DIGITS: [char; 16] = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
    ];
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(DIGITS[usize::from(byte >> 4)]);
        out.push(DIGITS[usize::from(byte & 0x0f)]);
    }
    out
}

/// Decodes a hex string, accepting both cases.
///
/// # Errors
///
/// Returns [`RemoteError::Invalid`] for an odd digit count or a non-hex digit.
pub(crate) fn hex_decode(text: &str) -> std::result::Result<Vec<u8>, RemoteError> {
    let text = text.trim();
    if text.len() % 2 != 0 {
        return Err(RemoteError::Invalid(
            "hex string has an odd number of digits".to_string(),
        ));
    }
    let mut out = Vec::with_capacity(text.len() / 2);
    for pair in text.as_bytes().chunks(2) {
        let high = hex_digit(pair[0])?;
        let low = hex_digit(pair[1])?;
        out.push((high << 4) | low);
    }
    Ok(out)
}

/// Standard padded base64 encoding.
pub(crate) fn base64_encode(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Standard padded base64 decoding.
///
/// # Errors
///
/// Returns [`RemoteError::Invalid`] when the text is not valid base64.
pub(crate) fn base64_decode(text: &str) -> std::result::Result<Vec<u8>, RemoteError> {
    base64::engine::general_purpose::STANDARD
        .decode(text.trim())
        .map_err(|error| RemoteError::Invalid(format!("invalid base64 encoding: {error}")))
}

/// One hex digit as a nibble.
fn hex_digit(byte: u8) -> std::result::Result<u8, RemoteError> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(RemoteError::Invalid(
            "hex string contains a non-hex digit".to_string(),
        )),
    }
}

/// Strips trailing slashes and adds `http://` when no scheme is present.
fn normalize_base_url(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    if trimmed.contains("://") {
        trimmed.to_string()
    } else {
        format!("http://{trimmed}")
    }
}

/// Whether a status is 2xx.
fn is_success(status: u16) -> bool {
    (200..=299).contains(&status)
}

/// Percent-encodes one query key or value.
fn percent_encode(value: &str) -> String {
    url::form_urlencoded::byte_serialize(value.as_bytes()).collect()
}

/// Builds the mapped error for a non-2xx response.
fn status_error(status: u16, body: &[u8], operation: &str) -> RemoteError {
    let message = match response_detail(body) {
        Some(detail) => format!("HTTP {status} for {operation}: {detail}"),
        None => format!("HTTP {status} for {operation}"),
    };
    if status == 404 {
        RemoteError::NotFound(message)
    } else {
        RemoteError::Status { status, message }
    }
}

/// Extracts a human-readable detail from a provider error body.
///
/// Uses the first non-empty of `"error"`, `"detail"`, `"message"`; string
/// values are kept as-is and other values are JSON-serialized. Falls back to the
/// trimmed body text.
fn response_detail(body: &[u8]) -> Option<String> {
    if let Ok(value) = serde_json::from_slice::<serde_json::Value>(body) {
        if let Some(map) = value.as_object() {
            for key in ["error", "detail", "message"] {
                if let Some(found) = map.get(key) {
                    let detail = match found {
                        serde_json::Value::String(text) => text.clone(),
                        serde_json::Value::Null => String::new(),
                        other => other.to_string(),
                    };
                    if !detail.trim().is_empty() {
                        return Some(detail);
                    }
                }
            }
        } else if let serde_json::Value::String(text) = &value {
            if !text.trim().is_empty() {
                return Some(text.clone());
            }
        }
    }
    let text = String::from_utf8_lossy(body).trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn endpoint(base_url: &str) -> SandboxEndpoint {
        SandboxEndpoint {
            base_url: base_url.to_string(),
            sandbox_id: None,
            isolation_key: Some("session_pre_deploy_aio_s1".to_string()),
        }
    }

    fn http_for(server: &MockServer) -> SandboxHttp {
        SandboxHttp::new(
            "aio",
            &endpoint(&server.uri()),
            &SandboxGatewayConfig::default(),
        )
        .unwrap()
    }

    #[test]
    fn base_url_is_trimmed_and_scheme_defaulted() {
        let config = SandboxGatewayConfig::default();
        let http =
            SandboxHttp::new("aio", &endpoint(" http://127.0.0.1:8123/// "), &config).unwrap();
        assert_eq!(http.base_url(), "http://127.0.0.1:8123");
        assert_eq!(http.provider(), "aio");

        let http = SandboxHttp::new("aio", &endpoint("127.0.0.1:8123"), &config).unwrap();
        assert_eq!(http.base_url(), "http://127.0.0.1:8123");
    }

    #[test]
    fn base_url_falls_back_to_the_launcher_configuration() {
        let mut config = SandboxGatewayConfig::default();
        let launcher = openjiuwen_core::sys_operation::sandbox::config::SandboxLauncherConfig {
            base_url: Some(" 10.0.0.7:9000 ".to_string()),
            ..Default::default()
        };
        config.launcher_config = Some(launcher);
        let http = SandboxHttp::new("jiuwenbox", &endpoint(""), &config).unwrap();
        assert_eq!(http.base_url(), "http://10.0.0.7:9000");
    }

    #[test]
    fn missing_base_url_is_a_configuration_error() {
        let error = SandboxHttp::new(
            "yuanrong",
            &endpoint("   "),
            &SandboxGatewayConfig::default(),
        )
        .err()
        .unwrap();
        assert_eq!(error.kind(), ErrorKind::Config);
        assert_eq!(
            error.message(),
            "yuanrong provider requires endpoint.base_url"
        );
    }

    #[test]
    fn timeout_is_the_configured_number_of_seconds() {
        let config = SandboxGatewayConfig::default();
        assert_eq!(
            SandboxHttp::new("aio", &endpoint("localhost"), &config)
                .unwrap()
                .timeout(),
            Duration::from_secs(30)
        );
        let config = SandboxGatewayConfig {
            timeout_seconds: 0,
            ..SandboxGatewayConfig::default()
        };
        assert_eq!(
            SandboxHttp::new("aio", &endpoint("localhost"), &config)
                .unwrap()
                .timeout(),
            Duration::from_secs(1)
        );
    }

    #[test]
    fn auth_query_parameters_are_appended_and_percent_encoded() {
        let mut config = SandboxGatewayConfig::default();
        config
            .auth_query_params
            .insert("token".to_string(), "a b&c".to_string());
        config
            .auth_query_params
            .insert("version".to_string(), "1".to_string());
        let http = SandboxHttp::new("aio", &endpoint("http://host"), &config).unwrap();
        assert_eq!(
            http.url("/v1/files"),
            "http://host/v1/files?token=a+b%26c&version=1"
        );
        assert_eq!(
            http.url("/v1/files?x=2"),
            "http://host/v1/files?x=2&token=a+b%26c&version=1"
        );
    }

    #[test]
    fn no_auth_query_keeps_the_path_untouched() {
        let http = SandboxHttp::new(
            "aio",
            &endpoint("http://host"),
            &SandboxGatewayConfig::default(),
        )
        .unwrap();
        assert_eq!(http.url("/v1/files"), "http://host/v1/files");
    }

    #[test]
    fn bearer_token_is_added_unless_authorization_is_configured() {
        let http = SandboxHttp::new(
            "aio",
            &endpoint("http://host"),
            &SandboxGatewayConfig::default(),
        )
        .unwrap()
        .with_bearer_token(Some(" secret ".to_string()));
        let built = http.request(reqwest::Method::GET, "/x").build().unwrap();
        assert_eq!(
            built.headers().get("authorization").unwrap(),
            "Bearer secret"
        );

        let empty = SandboxHttp::new(
            "aio",
            &endpoint("http://host"),
            &SandboxGatewayConfig::default(),
        )
        .unwrap()
        .with_bearer_token(Some("   ".to_string()));
        let built = empty.request(reqwest::Method::GET, "/x").build().unwrap();
        assert!(built.headers().get("authorization").is_none());

        let mut config = SandboxGatewayConfig::default();
        config
            .auth_headers
            .insert("AUTHORIZATION".to_string(), "Basic abc".to_string());
        let configured = SandboxHttp::new("aio", &endpoint("http://host"), &config)
            .unwrap()
            .with_bearer_token(Some("secret".to_string()));
        let built = configured
            .request(reqwest::Method::GET, "/x")
            .build()
            .unwrap();
        assert_eq!(
            built.headers().get("authorization").unwrap(),
            "Basic abc",
            "a configured authorization header wins"
        );
    }

    #[tokio::test]
    async fn auth_headers_are_sent_on_every_request() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/status"))
            .and(header("x-api-key", "k-1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"ok": true})))
            .mount(&server)
            .await;
        let mut config = SandboxGatewayConfig::default();
        config
            .auth_headers
            .insert("x-api-key".to_string(), "k-1".to_string());
        let http = SandboxHttp::new("aio", &endpoint(&server.uri()), &config).unwrap();
        let value = http
            .send_value(
                http.request(reqwest::Method::GET, "/v1/status"),
                "read_file",
            )
            .await
            .unwrap();
        assert_eq!(value, json!({"ok": true}));
    }

    #[tokio::test]
    async fn send_json_decodes_the_payload() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/read"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({"content": "hi", "size": 2})),
            )
            .mount(&server)
            .await;
        let http = http_for(&server);
        let value: serde_json::Value = http
            .send_json(
                http.request(reqwest::Method::POST, "/v1/read")
                    .json(&json!({"path": "a.txt"})),
                "read_file",
            )
            .await
            .unwrap();
        assert_eq!(value["content"], "hi");
    }

    #[tokio::test]
    async fn not_found_maps_to_not_found_with_the_provider_detail() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/files/missing"))
            .respond_with(
                ResponseTemplate::new(404).set_body_json(json!({"detail": "no such file"})),
            )
            .mount(&server)
            .await;
        let http = http_for(&server);
        let error = http
            .send_empty(
                http.request(reqwest::Method::GET, "/v1/files/missing"),
                "read_file",
            )
            .await
            .unwrap_err();
        assert!(error.is_not_found());
        assert!(!error.is_container_gone());
        assert_eq!(error.status(), Some(404));
        assert_eq!(error.message(), "HTTP 404 for read_file: no such file");
    }

    #[tokio::test]
    async fn other_statuses_carry_the_status_and_detail() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/write"))
            .respond_with(ResponseTemplate::new(503).set_body_json(json!({"error": {"code": 7}})))
            .mount(&server)
            .await;
        let http = http_for(&server);
        let error = http
            .send_value(
                http.request(reqwest::Method::POST, "/v1/write"),
                "write_file",
            )
            .await
            .unwrap_err();
        assert!(!error.is_not_found());
        assert_eq!(error.status(), Some(503));
        assert_eq!(error.message(), r#"HTTP 503 for write_file: {"code":7}"#);
    }

    #[tokio::test]
    async fn a_plain_text_body_is_the_detail() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/boom"))
            .respond_with(ResponseTemplate::new(500).set_body_string("  upstream exploded \n"))
            .mount(&server)
            .await;
        let http = http_for(&server);
        let error = http
            .send_empty(http.request(reqwest::Method::GET, "/v1/boom"), "list_files")
            .await
            .unwrap_err();
        assert_eq!(
            error.message(),
            "HTTP 500 for list_files: upstream exploded"
        );

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/empty"))
            .respond_with(ResponseTemplate::new(502))
            .mount(&server)
            .await;
        let http = http_for(&server);
        let error = http
            .send_empty(
                http.request(reqwest::Method::GET, "/v1/empty"),
                "list_files",
            )
            .await
            .unwrap_err();
        assert_eq!(error.message(), "HTTP 502 for list_files");
    }

    #[tokio::test]
    async fn a_non_json_body_on_success_is_a_transport_error() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/html"))
            .respond_with(ResponseTemplate::new(200).set_body_string("<html></html>"))
            .mount(&server)
            .await;
        let http = http_for(&server);
        let error = http
            .send_value(http.request(reqwest::Method::GET, "/v1/html"), "read_file")
            .await
            .unwrap_err();
        assert!(matches!(error, RemoteError::Transport(_)));
        assert!(error.message().contains("could not decode"));
    }

    #[tokio::test]
    async fn an_unreachable_server_is_a_transport_error() {
        // Bind then drop a listener so the port is (almost certainly) closed.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        let http = SandboxHttp::new(
            "aio",
            &endpoint(&format!("http://{address}")),
            &SandboxGatewayConfig::default(),
        )
        .unwrap();
        let error = http
            .send_empty(http.request(reqwest::Method::GET, "/v1/x"), "read_file")
            .await
            .unwrap_err();
        assert!(matches!(error, RemoteError::Transport(_)));
        assert!(
            !error.message().contains("token="),
            "the request URL is stripped from the message"
        );
    }

    #[tokio::test]
    async fn send_bytes_and_send_raw_keep_the_body_verbatim() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/blob"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(vec![0_u8, 1, 2, 255]))
            .mount(&server)
            .await;
        let http = http_for(&server);
        let bytes = http
            .send_bytes(http.request(reqwest::Method::GET, "/v1/blob"), "read_file")
            .await
            .unwrap();
        assert_eq!(bytes, vec![0_u8, 1, 2, 255]);
    }

    #[tokio::test]
    async fn send_raw_leaves_status_handling_to_the_provider() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/gone"))
            .respond_with(
                ResponseTemplate::new(404).set_body_json(json!({"error": "container gone"})),
            )
            .mount(&server)
            .await;
        let http = http_for(&server);
        let raw = http
            .send_raw(http.request(reqwest::Method::GET, "/v1/gone"), "read_file")
            .await
            .unwrap();
        assert_eq!(raw.status, 404);
        assert!(!raw.is_success());
        assert_eq!(raw.json().unwrap(), json!({"error": "container gone"}));
        assert!(raw.text().contains("container gone"));

        let empty = RawResponse {
            status: 204,
            body: Vec::new(),
        };
        assert!(empty.is_success());
        assert!(empty.json().is_none());
        assert_eq!(empty.text(), "");
    }

    #[test]
    fn gateway_failure_uses_the_sandbox_gateway_status() {
        let failure = gateway_failure::<u8>(
            "read_file",
            &RemoteError::NotFound("HTTP 404 for read_file: gone".to_string()),
        );
        assert_eq!(failure.code, 199_007);
        assert_eq!(
            failure.message,
            "sandbox gateway error, operation: read_file, reason: HTTP 404 for read_file: gone"
        );
        assert!(failure.data.is_none());
    }

    #[test]
    fn remote_error_maps_to_the_matching_error_kind() {
        let not_found = remote_error("read_file", &RemoteError::NotFound("gone".to_string()));
        assert_eq!(not_found.kind(), ErrorKind::NotFound);
        assert_eq!(not_found.message(), "read_file: gone");

        let invalid = remote_error("write_file", &RemoteError::Invalid("bad".to_string()));
        assert_eq!(invalid.kind(), ErrorKind::Config);
        assert_eq!(invalid.message(), "write_file: bad");

        for error in [
            RemoteError::ContainerGone("container gone".to_string()),
            RemoteError::AlreadyExists("exists".to_string()),
            RemoteError::SearchLimit("too many".to_string()),
            RemoteError::Transport("down".to_string()),
            RemoteError::Status {
                status: 500,
                message: "HTTP 500 for read_file".to_string(),
            },
        ] {
            assert_eq!(remote_error("read_file", &error).kind(), ErrorKind::SysOp);
        }
    }

    #[test]
    fn status_message_includes_the_code_when_it_is_missing() {
        let error = RemoteError::Status {
            status: 429,
            message: "slow down".to_string(),
        };
        assert_eq!(error.message(), "HTTP 429: slow down");
        assert_eq!(error.status(), Some(429));
    }

    #[test]
    fn hex_round_trips_and_rejects_bad_input() {
        assert_eq!(hex_encode(&[0x00, 0x0f, 0xa5, 0xff]), "000fa5ff");
        assert_eq!(
            hex_decode("000FA5ff").unwrap(),
            vec![0x00_u8, 0x0f, 0xa5, 0xff]
        );
        assert_eq!(hex_encode(&[]), "");
        assert_eq!(hex_decode("").unwrap(), Vec::<u8>::new());
        assert!(hex_decode("abc").is_err());
        assert!(hex_decode("zz").is_err());
        let error = hex_decode("g0").unwrap_err();
        assert!(matches!(error, RemoteError::Invalid(_)));
    }

    #[test]
    fn base64_round_trips_and_rejects_bad_input() {
        assert_eq!(base64_encode(b"hello"), "aGVsbG8=");
        assert_eq!(base64_decode("aGVsbG8=").unwrap(), b"hello".to_vec());
        assert_eq!(base64_encode(b""), "");
        assert!(base64_decode("not valid base64!").is_err());
        assert!(matches!(
            base64_decode("!!!!").unwrap_err(),
            RemoteError::Invalid(_)
        ));
    }
}
