//! Device-code login, token exchange, and refresh for the `OpenAI` account provider.
//!
//! The flow is the standard OAuth 2.0 device authorization grant against
//! `https://auth.openai.com`:
//!
//! 1. [`OpenAiOAuthClient::request_device_code`] asks for a user code.
//! 2. The caller shows [`DeviceCode::user_code`]; [`OpenAiOAuthClient::poll_device_token`]
//!    waits until the user has signed in and then exchanges the authorization
//!    code for tokens.
//! 3. [`OpenAiOAuthClient::access_token`] hands out a usable bearer token,
//!    refreshing it through [`OpenAiOAuthClient::refresh`] when it expires within
//!    `REFRESH_SKEW` and persisting the rotated refresh
//!    token through the [`TokenStore`].
//!
//! Every request goes through the shared [`HttpClient`], every error carries the
//! status and a redacted, truncated body, and no token, authorization code, code
//! verifier, or refresh token is ever logged or formatted into a message.

use std::time::Duration;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::{HttpClient, HttpRequest, HttpResponse};
use openjiuwen_common::security::{redact_str, SecretString};
use serde_json::{json, Value};

use crate::llm::transport::join_url;

use super::store::{parse_token_payload, DeviceCode, OAuthToken, TokenStore, REFRESH_SKEW};

/// Default OAuth issuer for the `OpenAI` account provider.
pub const DEFAULT_ISSUER: &str = "https://auth.openai.com";

/// Default OAuth client id used by the device-authorization grant.
pub const DEFAULT_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";

/// Path of the device-login callback the token exchange reports as `redirect_uri`.
pub const DEVICE_CALLBACK_PATH: &str = "/deviceauth/callback";

/// Path of the page a user opens to enter a device user code.
///
/// The device-code response carries only the code, so the verification URL is
/// derived from the configured issuer (Python parity:
/// `OPENAI_ACCOUNT_DEVICE_AUTH_URL = f"{issuer}/codex/device"`).
pub const DEVICE_VERIFICATION_PATH: &str = "/codex/device";

/// Path that hands out a device user code.
const USER_CODE_PATH: &str = "/api/accounts/deviceauth/usercode";

/// Path polled until the user completes the device login.
const DEVICE_TOKEN_PATH: &str = "/api/accounts/deviceauth/token";

/// Path of the OAuth token endpoint (code exchange and refresh).
const OAUTH_TOKEN_PATH: &str = "/oauth/token";

/// Shortest delay between two device-authorization polls.
const MIN_POLL_INTERVAL: Duration = Duration::from_secs(1);

/// Upper bound for the growing delay between two pending polls.
const MAX_POLL_BACKOFF: Duration = Duration::from_secs(10);

/// Overall polling budget used when the device-code response omits `expires_in`.
const DEFAULT_POLL_TIMEOUT: Duration = Duration::from_secs(15 * 60);

/// Maximum number of characters of a provider body kept in an error message.
const MAX_BODY_CHARS: usize = 500;

/// Marker appended to a body that had to be truncated.
const TRUNCATION_MARKER: char = '…';

/// Client for the `OpenAI` account OAuth endpoints.
///
/// The client is cheap to clone (it shares the [`HttpClient`] handle) and holds
/// no token cache: the [`TokenStore`] is the only source of truth, so several
/// clients — or several processes — never disagree about the current token.
#[derive(Debug, Clone)]
pub struct OpenAiOAuthClient {
    http: HttpClient,
    client_id: String,
    base_url: String,
}

impl OpenAiOAuthClient {
    /// Builds a client with the default issuer and client id.
    ///
    /// # Errors
    ///
    /// Returns the error of [`HttpClient::with_defaults`] when the HTTP stack
    /// cannot be built.
    pub fn new() -> Result<Self> {
        Ok(Self {
            http: HttpClient::with_defaults()?,
            client_id: DEFAULT_CLIENT_ID.to_owned(),
            base_url: DEFAULT_ISSUER.to_owned(),
        })
    }

    /// Replaces the HTTP client (retry policy, timeouts, proxies, TLS).
    #[must_use]
    pub fn with_http_client(mut self, http: HttpClient) -> Self {
        self.http = http;
        self
    }

    /// Replaces the OAuth issuer, for example with a loopback mock server.
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        let base_url = base_url.into();
        base_url
            .trim()
            .trim_end_matches('/')
            .clone_into(&mut self.base_url);
        self
    }

    /// Replaces the OAuth client id.
    #[must_use]
    pub fn with_client_id(mut self, client_id: impl Into<String>) -> Self {
        self.client_id = client_id.into();
        self
    }

    /// Returns the HTTP client every request goes through.
    #[must_use]
    pub fn http_client(&self) -> &HttpClient {
        &self.http
    }

    /// Returns the OAuth issuer.
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Returns the OAuth client id.
    #[must_use]
    pub fn client_id(&self) -> &str {
        &self.client_id
    }

    /// Requests a user code for the device-authorization grant.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Llm`] for a non-2xx response (the message carries the
    /// status and a redacted, truncated body) or a response that is not a
    /// [`DeviceCode`], and the transport error of the shared HTTP client when the
    /// request never completes.
    pub async fn request_device_code(&self) -> Result<DeviceCode> {
        let url = join_url(&self.base_url, USER_CODE_PATH);
        let body = json!({ "client_id": self.client_id });
        let response = self
            .http
            .execute(HttpRequest::post(url.clone()).json(&body)?)
            .await?;
        if !response.is_success() {
            return Err(operation_failed("device code request", &url, &response));
        }
        serde_json::from_slice::<DeviceCode>(response.bytes()).map_err(|error| {
            Error::llm(format!(
                "OpenAI account device code response from {url} is not a device code: {error}"
            ))
        })
    }

    /// Polls until the user finishes the device login and returns the tokens.
    ///
    /// Polling starts after one `interval` (at least one second) and backs off
    /// exponentially, capped at ten seconds, while the endpoint answers
    /// `403`/`404` ("still pending"). The whole flow is bounded by the device
    /// code's `expires_in`, and never polls longer than fifteen minutes.
    ///
    /// A `200` response carrying `authorization_code`/`code_verifier` is exchanged
    /// through [`OpenAiOAuthClient::exchange_code`]; a `200` that already carries
    /// a token is returned as-is.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Llm`] when the device code expires, when the endpoint
    /// rejects the poll (rate limiting included), or when an exchange fails, and
    /// the transport error of the shared HTTP client on a connection failure.
    pub async fn poll_device_token(&self, code: &DeviceCode) -> Result<OAuthToken> {
        let advertised = if code.expires_in > 0 {
            Duration::from_secs(code.expires_in)
        } else {
            DEFAULT_POLL_TIMEOUT
        };
        self.poll_device_token_with_deadline(code, advertised.min(DEFAULT_POLL_TIMEOUT))
            .await
    }

    /// [`OpenAiOAuthClient::poll_device_token`] with an explicit polling budget.
    ///
    /// The deadline exists so a caller (or a test) can bound the wait without
    /// mutating the device code; the production entry point derives it from
    /// `expires_in`.
    ///
    /// # Errors
    ///
    /// The errors of [`OpenAiOAuthClient::poll_device_token`].
    pub async fn poll_device_token_with_deadline(
        &self,
        code: &DeviceCode,
        deadline: Duration,
    ) -> Result<OAuthToken> {
        let started = tokio::time::Instant::now();
        let mut delay = poll_interval(code.interval);
        loop {
            let remaining = deadline.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                break;
            }
            tokio::time::sleep(delay.min(remaining)).await;

            let url = join_url(&self.base_url, DEVICE_TOKEN_PATH);
            let body = json!({
                "device_auth_id": code.device_auth_id,
                "user_code": code.user_code,
            });
            let response = self
                .http
                .execute(HttpRequest::post(url.clone()).json(&body)?)
                .await?;
            match response.status {
                200 => return self.complete_authorization(&url, response.bytes()).await,
                403 | 404 => {
                    tracing::debug!(
                        status = response.status,
                        "OpenAI account device authorization still pending"
                    );
                    delay = (delay * 2).min(MAX_POLL_BACKOFF);
                }
                _ => return Err(operation_failed("device auth polling", &url, &response)),
            }
        }
        Err(Error::llm(format!(
            "OpenAI account device authorization expired after {deadline:?} without a sign-in; \
             start a new device login"
        )))
    }

    /// Exchanges a device authorization code for tokens.
    ///
    /// `redirect_uri` must match the URI the user code was requested for; the
    /// device flow uses [`DEVICE_CALLBACK_PATH`] under the configured issuer.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Llm`] for a non-2xx response (redacted message) or a
    /// response that is not a token, and the transport error of the shared HTTP
    /// client on a connection failure.
    pub async fn exchange_code(
        &self,
        code: &str,
        code_verifier: &str,
        redirect_uri: &str,
    ) -> Result<OAuthToken> {
        let url = join_url(&self.base_url, OAUTH_TOKEN_PATH);
        let body = form_body(&[
            ("grant_type", "authorization_code"),
            ("code", code),
            ("redirect_uri", redirect_uri),
            ("client_id", &self.client_id),
            ("code_verifier", code_verifier),
        ]);
        self.post_token(&url, body, "token exchange", None).await
    }

    /// Mints a new access token from `token`'s refresh token.
    ///
    /// The request is exactly what the reference implementation sends — a
    /// form-encoded `POST {issuer}/oauth/token` carrying `grant_type`,
    /// `refresh_token`, and `client_id`, with no `scope` and no `redirect_uri`.
    ///
    /// The response may rotate the refresh token; the rotated value is honoured,
    /// and the previous one is kept when the response omits it. The same rule
    /// applies to `id_token`, `scope`, and `token_type`, and the expiry is
    /// resolved as `expires_at` → `expires_in` → the previous `expires_at` → the
    /// JWT `exp` claim.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when `token` carries no refresh token — the
    /// caller has to run the device-authorization flow again — and otherwise the
    /// errors of the token endpoint.
    pub async fn refresh(&self, token: &OAuthToken) -> Result<OAuthToken> {
        let refresh_token = token
            .refresh_token
            .as_ref()
            .map(SecretString::expose_secret)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                Error::config(
                    "OpenAI account OAuth token has no refresh token; run the device \
                     authorization flow to sign in again",
                )
            })?;
        let url = join_url(&self.base_url, OAUTH_TOKEN_PATH);
        let body = form_body(&[
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh_token),
            ("client_id", &self.client_id),
        ]);
        self.post_token(&url, body, "token refresh", Some(token))
            .await
    }

    /// Returns a usable access token, refreshing and persisting it when needed.
    ///
    /// The token is refreshed when `force_refresh` is set or when
    /// [`OAuthToken::is_expired`] reports it stale within [`REFRESH_SKEW`] — which
    /// includes a token whose expiry cannot be resolved, because such a credential
    /// has no known lifetime and must be replaced instead of used indefinitely. The
    /// (possibly rotated) result is saved through `store` before it is returned.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when the store holds no token (the message names
    /// the device-authorization flow to run), and the errors of
    /// [`OpenAiOAuthClient::refresh`] or of the store when the refresh fails.
    pub async fn access_token(
        &self,
        store: &dyn TokenStore,
        force_refresh: bool,
    ) -> Result<OAuthToken> {
        let token = store.load().await?.ok_or_else(|| {
            Error::config(
                "no OpenAI account OAuth token stored; run the device authorization flow \
                 (request_device_code + poll_device_token) first",
            )
        })?;
        if !force_refresh && !token.is_expired(REFRESH_SKEW) {
            return Ok(token);
        }
        let refreshed = self.refresh(&token).await?;
        store.save(&refreshed).await?;
        Ok(refreshed)
    }

    /// Turns a completed device authorization into tokens.
    async fn complete_authorization(&self, url: &str, payload: &[u8]) -> Result<OAuthToken> {
        let payload: Value = serde_json::from_slice(payload).map_err(|error| {
            Error::llm(format!(
                "OpenAI account device authorization response from {url} is not valid JSON: \
                 {error}"
            ))
        })?;
        let authorization_code = string_field(&payload, "authorization_code");
        let code_verifier = string_field(&payload, "code_verifier");
        match (authorization_code, code_verifier) {
            (Some(code), Some(code_verifier)) => {
                let redirect_uri = join_url(&self.base_url, DEVICE_CALLBACK_PATH);
                self.exchange_code(&code, &code_verifier, &redirect_uri)
                    .await
            }
            (None, None) => serde_json::from_value::<OAuthToken>(payload).map_err(|error| {
                Error::llm(format!(
                    "OpenAI account device authorization response from {url} carries neither \
                     authorization data nor a token: {error}"
                ))
            }),
            _ => Err(Error::llm(format!(
                "OpenAI account device authorization response from {url} is missing the \
                 authorization code or the code verifier"
            ))),
        }
    }

    /// Posts a form-encoded token request and parses the token response.
    ///
    /// `previous` is the token the response replaces (the refresh path) or
    /// `None` (the device-code exchange), and drives the field carry-over
    /// described on [`OAuthToken`].
    async fn post_token(
        &self,
        url: &str,
        body: String,
        operation: &str,
        previous: Option<&OAuthToken>,
    ) -> Result<OAuthToken> {
        let request = HttpRequest::post(url.to_owned())
            .header("content-type", "application/x-www-form-urlencoded")
            .header("accept", "application/json")
            .text(body);
        let response = self.http.execute(request).await?;
        if !response.is_success() {
            return Err(operation_failed(operation, url, &response));
        }
        parse_token_payload(response.bytes(), previous).map_err(|error| {
            // A successful status carrying an unusable token is a credential
            // problem, not a transport one: the reference implementation marks
            // every such case `relogin_required`.
            Error::llm(format!(
                "OpenAI account {operation} response from {url} is not a token ({error}); \
                 a re-login is required — sign in again with the device authorization flow"
            ))
        })
    }
}

/// Returns the polling delay for an advertised interval, enforcing the minimum.
fn poll_interval(interval: u64) -> Duration {
    Duration::from_secs(interval).max(MIN_POLL_INTERVAL)
}

/// Encodes `fields` as an `application/x-www-form-urlencoded` body.
fn form_body(fields: &[(&str, &str)]) -> String {
    let mut serializer = url::form_urlencoded::Serializer::new(String::new());
    for (key, value) in fields {
        serializer.append_pair(key, value);
    }
    serializer.finish()
}

/// Reads a non-blank string field out of a JSON object.
fn string_field(payload: &Value, key: &str) -> Option<String> {
    payload
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

/// Vendor error codes that invalidate the stored credentials.
///
/// The reference implementation (`_raise_oauth_failed`) marks exactly these codes
/// — plus any `401`/`403` status — as `relogin_required`.
const RELOGIN_CODES: [&str; 4] = [
    "invalid_grant",
    "invalid_token",
    "invalid_request",
    "refresh_token_reused",
];

/// Maps a rejected OAuth request to an error that cannot leak a credential.
///
/// Three outcomes are distinguished through the message, because the error
/// taxonomy is shared with the rest of the crate:
///
/// - `429`: rate limited — the stored credentials are still valid and the caller
///   should retry, not sign in again;
/// - `401`/`403`, or a vendor code in [`RELOGIN_CODES`]: the credentials are no
///   longer usable and the caller must run the device-authorization flow again;
/// - anything else: a plain provider failure, retryable or not.
///
/// The body itself is never echoed (only the shared redaction placeholder), and a
/// vendor code is echoed only when it is one of the known [`RELOGIN_CODES`], so an
/// arbitrary provider string cannot smuggle a secret into a log line.
fn operation_failed(operation: &str, url: &str, response: &HttpResponse) -> Error {
    let body = response.text();
    let detail = redacted_snippet(&body);
    let suffix = if detail.is_empty() {
        String::new()
    } else {
        format!(": {detail}")
    };
    if response.status == 429 {
        return Error::llm(format!(
            "OpenAI account {operation} is rate limited (HTTP 429) for {url}; \
             the stored credentials are still valid and no re-login is required{suffix}"
        ));
    }
    let code = vendor_error_code(&body);
    let known = code.as_deref().filter(|code| RELOGIN_CODES.contains(code));
    if known.is_some() || matches!(response.status, 401 | 403) {
        let reason = known.map_or_else(String::new, |code| format!(" ({code})"));
        return Error::llm(format!(
            "OpenAI account {operation} failed with HTTP {} for {url}{reason}; the stored \
             credentials are no longer usable and a re-login is required — sign in again with \
             the device authorization flow{suffix}",
            response.status
        ));
    }
    Error::llm(format!(
        "OpenAI account {operation} failed with HTTP {} for {url}{suffix}",
        response.status
    ))
}

/// Reads the vendor error code out of an OAuth error body.
///
/// The accepted shapes match the reference implementation: `{"error": {"code":
/// …}}`, `{"error": {"type": …}}`, and `{"error": "<code>"}`. A body that is not
/// JSON, or carries no `error` member in one of those shapes, yields `None`.
fn vendor_error_code(body: &str) -> Option<String> {
    let payload: Value = serde_json::from_str(body).ok()?;
    let code = match payload.get("error")? {
        Value::Object(error) => error.get("code").or_else(|| error.get("type"))?.as_str()?,
        Value::String(error) => error.as_str(),
        _ => return None,
    };
    let code = code.trim();
    (!code.is_empty()).then(|| code.to_owned())
}

/// Renders a provider body safe for an error message.
///
/// The body is redacted through
/// [`redact_str`](openjiuwen_common::security::redact_str) — a non-empty body
/// becomes the shared `<REDACTED>` placeholder — and truncated to
/// [`MAX_BODY_CHARS`] characters on a character boundary.
fn redacted_snippet(body: &str) -> String {
    let redacted = redact_str(body);
    let mut snippet: String = redacted.chars().take(MAX_BODY_CHARS).collect();
    if snippet.chars().count() < redacted.chars().count() {
        snippet.push(TRUNCATION_MARKER);
    }
    snippet
}
