//! `HTTP` moderation transport over the shared async client.
//!
//! [`HttpModerationTransport`] implements [`ModerationTransport`] for any
//! JSON-over-HTTP moderation endpoint: it posts `{"<text_field>": "<text>"}`,
//! attaches one credential, applies static headers, honours per-call timeouts,
//! retries transient statuses with exponential backoff, and maps every failure
//! to [`Error::Transport`] facts (service, status, bounded body excerpt) that
//! never repeat the scanned text.
//!
//! The crate-internal helpers `http_client`, `json_body`, and `resolve_timeout`
//! are shared with the vendor adapters, so client construction, status/JSON
//! error mapping, and timeout sanitising live in one place instead of being
//! repeated per vendor.
#![allow(clippy::module_name_repetitions)]

use std::collections::BTreeMap;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::{HttpClient, HttpClientConfig, HttpRequest, HttpResponse};
use openjiuwen_common::security::SecretString;

use super::api::{ModerationRequest, ModerationTransport};

/// Log target shared by every backend in this crate.
const LOG_TARGET: &str = "openjiuwen_security::backends";

/// Python's default request timeout (`timeout: float = 30.0`).
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Number of retries performed after the first attempt.
const DEFAULT_MAX_RETRIES: u32 = 2;

/// Initial retry backoff, matching the shared client's default of 200 ms.
const DEFAULT_RETRY_INITIAL_BACKOFF: Duration = Duration::from_millis(200);

/// Upper bound on the exponential retry backoff.
const RETRY_MAX_BACKOFF: Duration = Duration::from_secs(5);

/// JSON field carrying the scanned text (Python `{"text": ...}`).
const DEFAULT_TEXT_FIELD: &str = "text";

/// Maximum number of response-body characters quoted in an error message.
const ERROR_SNIPPET_CHARS: usize = 200;

/// Service label used when a URL carries no host component.
const UNKNOWN_SERVICE: &str = "moderation";

/// How the credential is attached to every moderation request.
///
/// `Bearer` produces an `Authorization: Bearer <secret>` header, `Header` sends
/// the secret verbatim under a vendor-specific name (Azure's `api-key`, for
/// example). The secret is held in a [`SecretString`], so `Debug` and logs show
/// a redaction marker while serialization keeps the plaintext for config
/// round-trips.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HttpAuth {
    /// No credential is attached by the transport itself.
    None,
    /// `Authorization: Bearer <token>`.
    Bearer(SecretString),
    /// A vendor-specific credential header.
    Header {
        /// Header name, for example `api-key`.
        name: String,
        /// Header value; redacted by [`SecretString`].
        value: SecretString,
    },
}

/// Configuration of the generic moderation transport.
///
/// Field names are `snake_case` and unknown JSON fields are tolerated, so a
/// configuration document written for one vendor adapter still loads. The
/// timeout and backoff fields keep Python's `f64` seconds; `resolve_timeout`
/// sanitises them before use.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct HttpModerationConfig {
    /// Absolute endpoint URL; a per-call [`ModerationRequest::url`] overrides it.
    pub url: String,
    /// Extra static headers sent with every request.
    pub headers: BTreeMap<String, String>,
    /// Credential attached to every request.
    pub auth: HttpAuth,
    /// Request timeout in seconds (`30.0` by default).
    pub timeout: f64,
    /// Retries after the first attempt (`2` by default; `0` sends one request).
    pub max_retries: u32,
    /// Backoff before the first retry, in seconds (`0.2` by default).
    pub retry_initial_backoff: f64,
    /// JSON field carrying the scanned text (`"text"` by default).
    pub text_field: String,
}

impl Default for HttpModerationConfig {
    /// The documented defaults: no URL, no headers, no credential, `30.0` s,
    /// two retries, a `0.2` s first backoff, and the `"text"` field.
    fn default() -> Self {
        Self {
            url: String::new(),
            headers: BTreeMap::new(),
            auth: HttpAuth::None,
            timeout: DEFAULT_TIMEOUT.as_secs_f64(),
            max_retries: DEFAULT_MAX_RETRIES,
            retry_initial_backoff: DEFAULT_RETRY_INITIAL_BACKOFF.as_secs_f64(),
            text_field: DEFAULT_TEXT_FIELD.to_owned(),
        }
    }
}

impl HttpModerationConfig {
    /// Sets the endpoint URL.
    #[must_use]
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = url.into();
        self
    }

    /// Adds one static header, replacing any previous value for that name.
    #[must_use]
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }

    /// Replaces the static headers.
    #[must_use]
    pub fn with_headers(mut self, headers: impl IntoIterator<Item = (String, String)>) -> Self {
        self.headers = headers.into_iter().collect();
        self
    }

    /// Sets the credential attached to every request.
    #[must_use]
    pub fn with_auth(mut self, auth: HttpAuth) -> Self {
        self.auth = auth;
        self
    }

    /// Sets the request timeout in seconds.
    #[must_use]
    pub fn with_timeout(mut self, seconds: f64) -> Self {
        self.timeout = seconds;
        self
    }

    /// Sets the number of retries after the first attempt.
    #[must_use]
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Sets the JSON field carrying the scanned text.
    #[must_use]
    pub fn with_text_field(mut self, text_field: impl Into<String>) -> Self {
        self.text_field = text_field.into();
        self
    }

    /// Rejects configurations the transport cannot honour.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for a blank URL or text field, a timeout
    /// that is not a finite positive number of seconds, a negative or
    /// non-finite backoff, an invalid header name/value, or an invalid
    /// credential header name/value.
    fn validate(&self) -> Result<()> {
        if self.url.trim().is_empty() {
            return Err(Error::validation(
                "moderation transport requires a non-empty endpoint URL",
            ));
        }
        if !self.timeout.is_finite() || self.timeout <= 0.0 {
            return Err(Error::validation(format!(
                "moderation timeout must be a finite positive number of seconds, got {}",
                self.timeout
            )));
        }
        if !self.retry_initial_backoff.is_finite() || self.retry_initial_backoff < 0.0 {
            return Err(Error::validation(format!(
                "moderation retry_initial_backoff must be a finite non-negative number of \
                 seconds, got {}",
                self.retry_initial_backoff
            )));
        }
        if self.text_field.trim().is_empty() {
            return Err(Error::validation("moderation text_field must not be blank"));
        }
        for (name, value) in &self.headers {
            validate_header_name(name)?;
            validate_header_value(name, value)?;
        }
        if let HttpAuth::Header { name, value } = &self.auth {
            validate_header_name(name)?;
            validate_header_value(name, value.expose_secret())?;
        }
        Ok(())
    }
}

/// Generic moderation transport speaking JSON over `HTTP`.
///
/// One transport owns one endpoint configuration and one shared client, so it
/// is cheap to clone and safe to share across guardrails and pipelines.
#[derive(Debug, Clone)]
pub struct HttpModerationTransport {
    config: HttpModerationConfig,
    client: HttpClient,
}

impl HttpModerationTransport {
    /// Builds a transport for `config`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the configuration cannot be honoured
    /// (blank URL, invalid header, non-positive timeout — see
    /// `HttpModerationConfig::validate`) and [`Error::Config`] when the
    /// underlying client cannot be built.
    pub fn new(config: HttpModerationConfig) -> Result<Self> {
        config.validate()?;
        let timeout = resolve_timeout(config.timeout);
        let backoff = resolve_backoff(config.retry_initial_backoff);
        let client = http_client(timeout, config.max_retries, backoff)?;
        tracing::debug!(
            target: LOG_TARGET,
            service = %service_name(&config.url),
            max_retries = config.max_retries,
            "moderation transport initialised"
        );
        Ok(Self { config, client })
    }

    /// Returns the configuration this transport was built from.
    #[must_use]
    pub fn config(&self) -> &HttpModerationConfig {
        &self.config
    }
}

#[async_trait]
impl ModerationTransport for HttpModerationTransport {
    /// Posts the request text to the resolved endpoint and returns the JSON reply.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when neither the request nor the configuration
    /// carries an endpoint, [`Error::Timeout`] when the deadline expires, and
    /// [`Error::Transport`] for connection failures, non-2xx statuses, and
    /// non-JSON bodies. Errors carry the service, status, and a bounded body
    /// excerpt — never the scanned text or a credential.
    async fn moderate(&self, request: &ModerationRequest) -> Result<Value> {
        let url = resolve_endpoint(&self.config.url, &request.url)?.to_owned();
        let service = service_name(&url);
        let timeout = if request.timeout.is_zero() {
            resolve_timeout(self.config.timeout)
        } else {
            request.timeout
        };

        let body = text_body(&self.config.text_field, &request.text);
        let mut builder = HttpRequest::post(url).json(&body)?.timeout(timeout);
        for (name, value) in &self.config.headers {
            builder = builder.header(name.clone(), value.clone());
        }
        builder = match &self.config.auth {
            HttpAuth::None => match &request.api_key {
                Some(key) => builder.bearer(key.clone()),
                None => builder,
            },
            HttpAuth::Bearer(secret) => builder.bearer(secret.clone()),
            HttpAuth::Header { name, value } => builder.header(name.clone(), value.expose_secret()),
        };

        let response = self.client.execute(builder).await?;
        let status = response.status;
        tracing::debug!(
            target: LOG_TARGET,
            service = %service,
            status,
            "moderation request completed"
        );
        json_body(response, &service)
    }
}

/// Builds the shared async client for one moderation endpoint.
///
/// Retries cover rate limiting (`429`) and every server error (`5xx`), with an
/// exponential backoff capped at [`RETRY_MAX_BACKOFF`]; `max_retries == 0`
/// disables retries entirely.
///
/// # Errors
///
/// Returns [`Error::Config`] when the client cannot be built with `timeout` and
/// the retry settings.
pub(crate) fn http_client(
    timeout: Duration,
    max_retries: u32,
    retry_initial_backoff: Duration,
) -> Result<HttpClient> {
    let config = HttpClientConfig::default()
        .with_timeout(timeout)
        .with_max_retries(max_retries)
        .with_retry_backoff(retry_initial_backoff, RETRY_MAX_BACKOFF)
        .with_retry_statuses(std::iter::once(429_u16).chain(500_u16..=599));
    HttpClient::new(config)
}

/// Validates the status of `response` and decodes its body as JSON.
///
/// `service` names the endpoint in diagnostics (normally the URL host); it is
/// never the scanned text. A non-2xx status becomes [`Error::Transport`]
/// carrying the status code and at most [`ERROR_SNIPPET_CHARS`] characters of
/// the whitespace-collapsed body, and a 2xx body that is not JSON becomes
/// [`Error::Transport`] naming only the body length and the parse failure.
///
/// # Errors
///
/// Returns [`Error::Transport`] for a non-2xx status or an undecodable body.
pub(crate) fn json_body(response: HttpResponse, service: &str) -> Result<Value> {
    let HttpResponse { status, body, .. } = response;
    if !(200..300).contains(&status) {
        tracing::warn!(
            target: LOG_TARGET,
            service,
            status,
            "moderation endpoint returned an error status"
        );
        return Err(Error::transport(format!(
            "moderation service {service} returned HTTP {status}: {}",
            body_snippet(&body)
        )));
    }
    serde_json::from_slice(&body).map_err(|err| {
        tracing::warn!(
            target: LOG_TARGET,
            service,
            status,
            "moderation endpoint returned a non-JSON body"
        );
        Error::transport(format!(
            "moderation service {service} returned a non-JSON body of {} bytes: {err}",
            body.len()
        ))
    })
}

/// Converts a Python-style timeout in seconds into a [`Duration`].
///
/// `NaN`, infinities, negative values, and zero fall back to 30 s, so callers
/// can pass an unvalidated `f64` without risking a panic.
#[must_use]
pub(crate) fn resolve_timeout(secs: f64) -> Duration {
    if !secs.is_finite() || secs <= 0.0 {
        return DEFAULT_TIMEOUT;
    }
    Duration::try_from_secs_f64(secs).unwrap_or(DEFAULT_TIMEOUT)
}

/// Converts the configured first-retry backoff into a [`Duration`].
///
/// Unusable values fall back to the 200 ms default rather than panicking.
fn resolve_backoff(secs: f64) -> Duration {
    if !secs.is_finite() || secs < 0.0 {
        return DEFAULT_RETRY_INITIAL_BACKOFF;
    }
    Duration::try_from_secs_f64(secs).unwrap_or(DEFAULT_RETRY_INITIAL_BACKOFF)
}

/// Picks the endpoint for one call: the non-blank per-call URL, else the
/// configured one.
///
/// # Errors
///
/// Returns [`Error::Validation`] when both URLs are blank.
fn resolve_endpoint<'a>(configured: &'a str, requested: &'a str) -> Result<&'a str> {
    let requested = requested.trim();
    let endpoint = if requested.is_empty() {
        configured.trim()
    } else {
        requested
    };
    if endpoint.is_empty() {
        return Err(Error::validation(
            "moderation transport received no endpoint URL",
        ));
    }
    Ok(endpoint)
}

/// Extracts the host of `url` as a diagnostic label, without userinfo, path,
/// or query string (a query may carry a credential).
fn service_name(url: &str) -> String {
    let without_scheme = url.split_once("://").map_or(url, |(_, rest)| rest);
    let authority = without_scheme
        .split(['/', '?', '#'])
        .next()
        .unwrap_or_default();
    let host = authority
        .rsplit_once('@')
        .map_or(authority, |(_, host)| host);
    if host.is_empty() {
        UNKNOWN_SERVICE.to_owned()
    } else {
        host.to_owned()
    }
}

/// Builds the `{"<field>": "<text>"}` request payload.
fn text_body(field: &str, text: &str) -> Value {
    let mut body = serde_json::Map::new();
    body.insert(field.to_owned(), Value::String(text.to_owned()));
    Value::Object(body)
}

/// Collapses the response body to a single line of at most
/// [`ERROR_SNIPPET_CHARS`] characters.
fn body_snippet(body: &[u8]) -> String {
    let collapsed = String::from_utf8_lossy(body)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if collapsed.is_empty() {
        return "(empty body)".to_owned();
    }
    let mut snippet: String = collapsed.chars().take(ERROR_SNIPPET_CHARS).collect();
    if collapsed.chars().count() > ERROR_SNIPPET_CHARS {
        snippet.push_str("...");
    }
    snippet
}

/// Rejects header names that are not RFC 7230 tokens.
fn validate_header_name(name: &str) -> Result<()> {
    if name.is_empty() || !name.bytes().all(is_token_byte) {
        return Err(Error::validation(format!(
            "invalid moderation header name {name:?}: expected an RFC 7230 token"
        )));
    }
    Ok(())
}

/// Rejects header values containing control characters (header injection).
fn validate_header_value(name: &str, value: &str) -> Result<()> {
    let invalid = value
        .bytes()
        .any(|byte| (byte < 0x20 && byte != b'\t') || byte == 0x7f);
    if invalid {
        return Err(Error::validation(format!(
            "invalid value for moderation header {name}"
        )));
    }
    Ok(())
}

/// Returns `true` for the ASCII bytes allowed in an RFC 7230 token.
fn is_token_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&byte)
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use openjiuwen_common::error::ErrorKind;

    use super::*;

    /// Builds a transport whose first retry backoff is 1 ms.
    fn transport_with(config: HttpModerationConfig) -> HttpModerationTransport {
        let config = HttpModerationConfig {
            retry_initial_backoff: 0.001,
            ..config
        };
        HttpModerationTransport::new(config).unwrap()
    }

    /// Configuration for the server's `/moderate` endpoint, retries disabled.
    fn config_for(server: &MockServer) -> HttpModerationConfig {
        HttpModerationConfig::default()
            .with_url(endpoint(server))
            .with_max_retries(0)
    }

    fn endpoint(server: &MockServer) -> String {
        format!("{}/moderate", server.uri())
    }

    /// Mounts a single-shot `POST /moderate` mock answering `response`.
    async fn server_with(response: ResponseTemplate) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/moderate"))
            .respond_with(response)
            .expect(1)
            .mount(&server)
            .await;
        server
    }

    fn ok_reply() -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_json(json!({"ok": true}))
    }

    /// A request with a blank per-call URL, so the configured endpoint is used.
    fn call(text: &str) -> ModerationRequest {
        ModerationRequest {
            url: String::new(),
            api_key: None,
            text: text.to_owned(),
            timeout: Duration::ZERO,
        }
    }

    fn header_value<'a>(request: &'a wiremock::Request, name: &str) -> Option<&'a str> {
        request.headers.get(name)?.to_str().ok()
    }

    fn body_json(request: &wiremock::Request) -> Value {
        serde_json::from_slice(&request.body).unwrap()
    }

    #[tokio::test]
    async fn posts_the_text_with_static_headers_and_a_bearer_credential() {
        let server = server_with(
            ResponseTemplate::new(200).set_body_json(json!({"flagged": true, "score": 0.9})),
        )
        .await;
        let config = config_for(&server)
            .with_header("X-Client", "openjiuwen")
            .with_auth(HttpAuth::Bearer(SecretString::from("sk-transport")));

        let reply = transport_with(config)
            .moderate(&call("please moderate this"))
            .await
            .unwrap();
        assert_eq!(reply, json!({"flagged": true, "score": 0.9}));

        let received = server.received_requests().await.unwrap();
        assert_eq!(received.len(), 1);
        let sent = &received[0];
        assert_eq!(sent.method.as_str(), "POST");
        assert_eq!(sent.url.path(), "/moderate");
        assert_eq!(body_json(sent), json!({"text": "please moderate this"}));
        assert_eq!(header_value(sent, "x-client"), Some("openjiuwen"));
        assert_eq!(
            header_value(sent, "authorization"),
            Some("Bearer sk-transport")
        );
        assert_eq!(header_value(sent, "content-type"), Some("application/json"));
    }

    #[tokio::test]
    async fn request_url_and_api_key_override_the_configuration() {
        let server = server_with(ok_reply()).await;
        let config = HttpModerationConfig::default()
            .with_url(format!("{}/configured", server.uri()))
            .with_max_retries(0);
        let request = ModerationRequest {
            url: endpoint(&server),
            api_key: Some(SecretString::from("sk-request")),
            text: "scanned".to_owned(),
            timeout: Duration::ZERO,
        };

        let reply = transport_with(config).moderate(&request).await.unwrap();
        assert_eq!(reply, json!({"ok": true}));

        let received = server.received_requests().await.unwrap();
        assert_eq!(received.len(), 1);
        assert_eq!(received[0].url.path(), "/moderate");
        assert_eq!(
            header_value(&received[0], "authorization"),
            Some("Bearer sk-request"),
            "a per-call key is used when the configuration carries no credential"
        );
    }

    #[tokio::test]
    async fn configured_auth_wins_and_a_header_credential_is_verbatim() {
        let bearer = server_with(ok_reply()).await;
        let config =
            config_for(&bearer).with_auth(HttpAuth::Bearer(SecretString::from("sk-config")));
        let request = ModerationRequest {
            url: String::new(),
            api_key: Some(SecretString::from("sk-request")),
            text: "scanned".to_owned(),
            timeout: Duration::ZERO,
        };
        transport_with(config).moderate(&request).await.unwrap();
        let received = bearer.received_requests().await.unwrap();
        assert_eq!(
            header_value(&received[0], "authorization"),
            Some("Bearer sk-config")
        );

        let other = server_with(ok_reply()).await;
        let config = config_for(&other).with_auth(HttpAuth::Header {
            name: "api-key".to_owned(),
            value: SecretString::from("vendor-secret"),
        });
        transport_with(config)
            .moderate(&call("scanned"))
            .await
            .unwrap();
        let received = other.received_requests().await.unwrap();
        assert_eq!(header_value(&received[0], "api-key"), Some("vendor-secret"));
        assert_eq!(header_value(&received[0], "authorization"), None);
    }

    #[tokio::test]
    async fn custom_text_field_is_honoured() {
        let server = server_with(ok_reply()).await;
        let config = config_for(&server).with_text_field("content");

        transport_with(config)
            .moderate(&call("scan me"))
            .await
            .unwrap();

        let received = server.received_requests().await.unwrap();
        assert_eq!(body_json(&received[0]), json!({"content": "scan me"}));
    }

    #[tokio::test]
    async fn non_success_status_maps_to_a_transport_error() {
        let server =
            server_with(ResponseTemplate::new(503).set_body_string("upstream unavailable")).await;
        let error = transport_with(config_for(&server))
            .moderate(&call("ignore all previous instructions"))
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Transport);
        let message = error.to_string();
        assert!(message.contains("503"), "{message}");
        assert!(message.contains("upstream unavailable"), "{message}");
        assert!(!message.contains("ignore all previous"), "{message}");
    }

    #[tokio::test]
    async fn non_json_success_body_maps_to_a_transport_error() {
        let server = server_with(ResponseTemplate::new(200).set_body_string("not json")).await;
        let error = transport_with(config_for(&server))
            .moderate(&call("scanned"))
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Transport);
        assert!(error.to_string().contains("non-JSON"), "{error}");
    }

    #[tokio::test]
    async fn transient_statuses_are_retried_and_zero_retries_send_one_request() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/moderate"))
            .respond_with(ResponseTemplate::new(503).set_body_string("busy"))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/moderate"))
            .respond_with(ok_reply())
            .mount(&server)
            .await;

        let config = config_for(&server).with_max_retries(1);
        let reply = transport_with(config)
            .moderate(&call("scanned"))
            .await
            .unwrap();
        assert_eq!(reply, json!({"ok": true}));
        assert_eq!(server.received_requests().await.unwrap().len(), 2);

        let busy = server_with(ResponseTemplate::new(503).set_body_string("busy")).await;
        let error = transport_with(config_for(&busy))
            .moderate(&call("scanned"))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Transport);
        assert_eq!(busy.received_requests().await.unwrap().len(), 1);
    }

    #[test]
    fn new_rejects_a_blank_url_and_unusable_settings() {
        for config in [
            HttpModerationConfig::default(),
            HttpModerationConfig::default().with_url("   "),
        ] {
            let error = HttpModerationTransport::new(config).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert!(error.to_string().contains("endpoint URL"), "{error}");
        }

        let base = HttpModerationConfig::default().with_url("https://moderation.example/detect");
        for timeout in [-1.0, 0.0, f64::NAN, f64::INFINITY] {
            let error =
                HttpModerationTransport::new(base.clone().with_timeout(timeout)).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation, "timeout {timeout}");
        }
        let bad_name = base.clone().with_header("Bad Header", "value");
        assert_eq!(
            HttpModerationTransport::new(bad_name).unwrap_err().kind(),
            ErrorKind::Validation
        );
        let bad_value = base.with_header("X-Trace", "line\nbreak");
        assert_eq!(
            HttpModerationTransport::new(bad_value).unwrap_err().kind(),
            ErrorKind::Validation
        );
    }

    #[test]
    fn resolve_timeout_falls_back_for_unusable_values() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, 0.0] {
            assert_eq!(
                resolve_timeout(bad),
                Duration::from_secs(30),
                "timeout {bad} must fall back to 30 s"
            );
        }
        assert_eq!(resolve_timeout(2.5), Duration::from_secs_f64(2.5));
        assert_eq!(resolve_timeout(0.001), Duration::from_millis(1));
    }

    #[test]
    fn endpoint_resolution_prefers_the_request_and_rejects_blanks() {
        assert_eq!(
            resolve_endpoint("https://cfg.example/a", " https://req.example/b ").unwrap(),
            "https://req.example/b"
        );
        assert_eq!(
            resolve_endpoint(" https://cfg.example/a ", "   ").unwrap(),
            "https://cfg.example/a",
            "a blank per-call URL keeps the configured endpoint"
        );
        assert_eq!(
            resolve_endpoint("  ", "  ").unwrap_err().kind(),
            ErrorKind::Validation
        );
    }

    #[test]
    fn service_name_drops_userinfo_and_paths() {
        assert_eq!(
            service_name("https://user:secret@moderation.example:8443/v1/detect?api_key=x"),
            "moderation.example:8443"
        );
        assert_eq!(
            service_name("moderation.example/detect"),
            "moderation.example"
        );
        assert_eq!(service_name(""), UNKNOWN_SERVICE);
    }

    #[test]
    fn config_defaults_round_trip_and_hide_secrets() {
        let defaults = HttpModerationConfig::default();
        assert_eq!(defaults.url, "");
        assert!(defaults.headers.is_empty());
        assert_eq!(defaults.auth, HttpAuth::None);
        assert!((defaults.timeout - 30.0).abs() < f64::EPSILON);
        assert_eq!(defaults.max_retries, 2);
        assert!((defaults.retry_initial_backoff - 0.2).abs() < f64::EPSILON);
        assert_eq!(defaults.text_field, "text");

        let config = defaults
            .with_url("https://moderation.example/detect")
            .with_header("X-Tenant", "acme")
            .with_auth(HttpAuth::Bearer(SecretString::from("sk-super-secret")))
            .with_timeout(12.5)
            .with_max_retries(3)
            .with_text_field("content");

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"timeout\":12.5"), "{json}");
        assert!(json.contains("\"text_field\":\"content\""), "{json}");
        let restored: HttpModerationConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, config);

        let debug = format!("{config:?}");
        assert!(debug.contains("moderation.example"), "{debug}");
        assert!(!debug.contains("sk-super-secret"), "{debug}");

        let header_auth = HttpAuth::Header {
            name: "api-key".to_owned(),
            value: SecretString::from("vendor-secret"),
        };
        let debug = format!("{header_auth:?}");
        assert!(!debug.contains("vendor-secret"), "{debug}");
        let restored: HttpAuth =
            serde_json::from_str(&serde_json::to_string(&header_auth).unwrap()).unwrap();
        assert_eq!(restored, header_auth);
    }

    #[test]
    fn partial_and_unknown_json_fields_are_tolerated() {
        let config: HttpModerationConfig = serde_json::from_str(
            r#"{"url": "https://moderation.example/detect", "vendor": {"name": "azure"}}"#,
        )
        .unwrap();
        assert_eq!(config.url, "https://moderation.example/detect");
        assert_eq!(config.auth, HttpAuth::None);
        assert_eq!(config.max_retries, 2);
        assert_eq!(config.text_field, "text");
    }
}
