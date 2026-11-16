//! Minting, caching and refreshing the OAuth 2.0 access token of a service account.
//!
//! Google's JWT-bearer flow exchanges a signed assertion for a short-lived access token: the
//! assertion built by [`signing::assertion`](crate::object_artifact::gcs::signing) is posted as
//! a form field to the key file's `token_uri`, and the reply carries the token and its lifetime.
//!
//! [`TokenSource`] owns that exchange. It mints lazily — the first request that needs a token
//! mints one — and caches it until [`REFRESH_MARGIN`] before expiry, so a burst of object
//! operations signs one assertion rather than one per request. The cache sits behind a
//! [`tokio::sync::Mutex`] that is held across the mint, so two concurrent requests never mint
//! twice: the second one waits and then reuses the first one's token.
//!
//! Nothing here is ever logged or rendered: the assertion, the minted token, the PEM and the
//! exchange's form body stay out of every error message and out of [`Debug`].

use std::fmt;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use openjiuwen_common::http::{HttpClient, HttpRequest};
use openjiuwen_common::security::SecretString;
use serde_json::Value;

use super::key::ServiceAccountKey;
use crate::error::{Error, Result};

/// How long before expiry a cached token is replaced.
///
/// Re-minting one minute early keeps a token from expiring in the middle of a request while
/// staying far cheaper than a token per request.
pub(super) const REFRESH_MARGIN: Duration = Duration::from_secs(60);

/// Lifetime used when the token endpoint reports no `expires_in`.
///
/// Google's tokens last an hour and the field is documented, so this only covers a reply that
/// omits it.
const DEFAULT_EXPIRES_IN_SECONDS: u64 = 3600;

/// The grant type of the JWT-bearer flow (RFC 7523 §2.1).
const GRANT_TYPE: &str = "urn:ietf:params:oauth:grant-type:jwt-bearer";

/// Maximum number of response-body characters echoed in a token error message.
const ERROR_BODY_EXCERPT: usize = 200;

/// Uppercase hexadecimal digits of a percent escape.
const HEX_DIGITS: &[u8; 16] = b"0123456789ABCDEF";

/// A cached access token and the instant it stops being usable.
struct CachedToken {
    /// The bearer token itself.
    token: SecretString,
    /// The instant the token expires, as reported by the endpoint.
    expires_at: Instant,
}

/// A service-account key plus the access token minted from it.
///
/// One source is attached to one [`GcsObjectStore`](crate::object_artifact::GcsObjectStore) and
/// shared through an `Arc`.
pub(super) struct TokenSource {
    /// The key the assertion is signed with.
    key: ServiceAccountKey,
    /// The current token, when one has been minted and has not been refreshed yet.
    cache: tokio::sync::Mutex<Option<CachedToken>>,
}

impl TokenSource {
    /// Wraps `key`; no token is minted until the first request needs one.
    pub(super) fn new(key: ServiceAccountKey) -> Self {
        Self {
            key,
            cache: tokio::sync::Mutex::new(None),
        }
    }

    /// Returns a usable access token, minting (or refreshing) one when needed.
    ///
    /// The exchange is serialized through the cache's mutex, so concurrent callers share one
    /// mint. `timeout` bounds the token request; `client` is the store's shared client, so the
    /// token endpoint sees the same TLS, proxy and redirect policy as the object endpoints.
    ///
    /// # Errors
    ///
    /// [`Error::External`] when the assertion cannot be produced, the endpoint is unreachable,
    /// answers non-2xx, or answers with a document that carries no usable `access_token`. The
    /// message names the endpoint, never the assertion or a token.
    pub(super) async fn access_token(
        &self,
        client: &HttpClient,
        timeout: Duration,
    ) -> Result<SecretString> {
        let mut cache = self.cache.lock().await;
        let now = Instant::now();
        if let Some(cached) = cache.as_ref() {
            if now + REFRESH_MARGIN < cached.expires_at {
                return Ok(cached.token.clone());
            }
        }

        let assertion = super::signing::assertion(&self.key, unix_now())?;
        let body = format!(
            "grant_type={}&assertion={}",
            percent_encode(GRANT_TYPE),
            percent_encode(&assertion)
        );
        let request = HttpRequest::post(self.key.token_uri())
            .header("content-type", "application/x-www-form-urlencoded")
            .text(body)
            .timeout(timeout);
        let response = client.execute(request).await?;
        if !response.is_success() {
            return Err(Error::external(TokenExchangeFailure::new(
                self.key.token_uri(),
                response.status,
                &response.body,
                &assertion,
            )));
        }

        let document: TokenResponse = serde_json::from_slice(&response.body).map_err(|error| {
            Error::external(error).with_context(format!(
                "decoding the service-account token response from {}",
                self.key.token_uri()
            ))
        })?;
        let token = document.access_token.unwrap_or_default();
        if token.trim().is_empty() {
            return Err(Error::external(MissingAccessToken {
                url: self.key.token_uri().to_string(),
            }));
        }

        let lifetime = document.expires_in.unwrap_or(DEFAULT_EXPIRES_IN_SECONDS);
        let token = SecretString::new(token);
        *cache = Some(CachedToken {
            token: token.clone(),
            expires_at: Instant::now() + Duration::from_secs(lifetime),
        });
        Ok(token)
    }
}

impl fmt::Debug for TokenSource {
    /// Renders the public shape only: the cached token is reported as a flag, never printed.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cached = self.cache.try_lock().is_ok_and(|cache| cache.is_some());
        formatter
            .debug_struct("TokenSource")
            .field("token_uri", &self.key.token_uri())
            .field("cached", &cached)
            .finish()
    }
}

/// The token endpoint's reply, reduced to the two fields the source needs.
#[derive(Debug, Default)]
struct TokenResponse {
    /// The bearer token; absent only in a malformed reply.
    access_token: Option<String>,
    /// Lifetime of the token in seconds, as a number or a decimal string.
    expires_in: Option<u64>,
}

/// Reads `expires_in` from either a JSON number or a decimal string.
///
/// The endpoint documents a number, but some intermediaries render it as a string, so both
/// shapes are accepted; anything else means "use the default lifetime".
fn decode_expires_in(value: &Value) -> Option<u64> {
    match value {
        Value::Number(number) => number.as_u64(),
        Value::String(text) => text.trim().parse().ok(),
        _ => None,
    }
}

/// Deserializes a token reply, accepting `expires_in` in either JSON shape.
///
/// Everything else in the document (`token_type`, `scope`, …) is ignored, and a non-object
/// document still decodes — into an empty reply, which the caller rejects for its missing
/// `access_token` rather than for its shape.
impl<'de> serde::Deserialize<'de> for TokenResponse {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let document = <Value as serde::Deserialize>::deserialize(deserializer)?;
        let Some(object) = document.as_object() else {
            return Ok(Self::default());
        };
        Ok(Self {
            access_token: object
                .get("access_token")
                .and_then(Value::as_str)
                .map(str::to_string),
            expires_in: object.get("expires_in").and_then(decode_expires_in),
        })
    }
}

/// The token endpoint answered 2xx without an `access_token`.
#[derive(Debug)]
struct MissingAccessToken {
    /// The token endpoint that answered.
    url: String,
}

impl fmt::Display for MissingAccessToken {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "the service-account token endpoint {} answered without an `access_token`",
            self.url
        )
    }
}

impl std::error::Error for MissingAccessToken {}

/// A rejected token exchange, rendered without ever echoing the assertion or a token.
#[derive(Debug)]
struct TokenExchangeFailure {
    /// The token endpoint that rejected the assertion.
    url: String,
    /// HTTP status of the rejection.
    status: u16,
    /// OAuth error code (`invalid_grant`, …), when the reply names one.
    error: Option<String>,
    /// OAuth error description, when the reply names one.
    description: Option<String>,
    /// Body excerpt, used when the reply is not an OAuth error document.
    excerpt: String,
}

impl TokenExchangeFailure {
    /// Builds the failure from the endpoint, the status, the reply body and the assertion.
    ///
    /// `assertion` is scrubbed out of the rendered body and description: a proxy or a
    /// misconfigured endpoint may echo the request, and the assertion is a bearer credential
    /// for the token exchange.
    fn new(url: &str, status: u16, body: &[u8], assertion: &str) -> Self {
        let document: Option<Value> = serde_json::from_slice(body).ok();
        let object = document.as_ref().and_then(Value::as_object);
        let text = String::from_utf8_lossy(body);
        Self {
            url: url.to_string(),
            status,
            error: object.and_then(|object| oauth_error(object.get("error"))),
            description: object
                .and_then(|object| object.get("error_description"))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|description| !description.is_empty())
                .map(|description| scrub(description, assertion)),
            excerpt: scrub(&excerpt(&text), assertion),
        }
    }
}

impl fmt::Display for TokenExchangeFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let error = self.error.as_deref().unwrap_or("<no error code>");
        let description = self.description.as_deref().unwrap_or(self.excerpt.as_str());
        write!(
            formatter,
            "service-account token exchange failed with HTTP {} for {}: {}: {}",
            self.status, self.url, error, description
        )
    }
}

impl std::error::Error for TokenExchangeFailure {}

/// Reads an OAuth `error` member: a string, or an object carrying a `message`.
fn oauth_error(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::String(text) if !text.trim().is_empty() => Some(text.trim().to_string()),
        Value::Object(object) => object
            .get("message")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|message| !message.is_empty())
            .map(str::to_string),
        _ => None,
    }
}

/// Shortens a response body for an error message.
fn excerpt(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return "<empty body>".to_string();
    }
    let mut excerpt: String = trimmed.chars().take(ERROR_BODY_EXCERPT).collect();
    if trimmed.chars().count() > ERROR_BODY_EXCERPT {
        excerpt.push('…');
    }
    excerpt
}

/// Replaces every occurrence of the assertion — raw and percent-encoded — in `text`.
///
/// A rejected exchange must never publish the credential it presented, even when the endpoint
/// (or an intermediary) echoes the request in its error document.
fn scrub(text: &str, assertion: &str) -> String {
    if assertion.is_empty() {
        return text.to_string();
    }
    text.replace(assertion, "<redacted-assertion>")
        .replace(&percent_encode(assertion), "<redacted-assertion>")
}

/// Percent-encodes `text` for an `application/x-www-form-urlencoded` body.
///
/// Unreserved RFC 3986 characters stay as they are and every other byte becomes `%XX`, which is
/// exactly what a form field needs (a space would be `+`, but an assertion and the grant type
/// never contain one).
fn percent_encode(text: &str) -> String {
    let mut encoded = String::with_capacity(text.len());
    for byte in text.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(char::from(byte));
            }
            _ => {
                encoded.push('%');
                encoded.push(char::from(HEX_DIGITS[usize::from(byte >> 4)]));
                encoded.push(char::from(HEX_DIGITS[usize::from(byte & 0x0f)]));
            }
        }
    }
    encoded
}

/// The current Unix time in seconds, or `0` when the clock is before the epoch.
fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use base64::Engine as _;
    use openjiuwen_common::http::HttpClientConfig;
    use serde_json::json;
    use wiremock::matchers::{body_string_contains, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// The same 2048-bit PKCS#8 RSA key the other modules' tests use.
    const SERVICE_ACCOUNT_PEM: &str = "-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDWEwu1E3LIrY/1
ukZXqTj+8wVh/tbtgPgivj104Ox3FLpvj3jhRu3QRibNMN6DM4Fi13J6IE2WRdxF
qiwyVX9lg98zpl6msySu3EYewibFHWPyW0OaKzm1GqNjzvFev/O7CqvA6KJUtgfj
rneYOGBqpl//ojbdO7d9/kq1ivXmaM6lvx+JSssHnYbvdfmH3PtSGfdtODuNSrrI
yDyAycSKmi6DIxojI3hJGXZm9oMX0Dh39sXOvTk+s66b59cDqq2TZfFoUD4Z9wtj
7b34kAZj4pGKk36OT+TDXzmypKvFbf7lHYSf+P4RN9EasunktpoRSQRlpXElNkKd
qgkZft/hAgMBAAECggEAds7TLIfCqIVK5EGhUH1mWHhfgGeCuQVesUmLqhY6pkwm
vEuP9tb6nSCR4EdC9uXa5RRVQFo1NTVIbpFnPfgdYCWz9mJbHYFG1pUBhYiTI8qP
8a2c8y8SM+S40WfEpaU6i4/8im+RyBnOfxoQ3bZMLGRx2toYQOgnihFbPS4GEv0L
8NTdj0XGNw09cy9ckIosHaZjF7MvNleeJjSR0l4Bl6HbCAS8DAxmrwXzEX6tRXdp
rYrjkqSP6qvC4+S7L4q6OpPNqRSl4urTC9ubFhtyfmo3FlMBC9w7uHLEghyfbSAF
WrHYjTxgRZnifeb9RfPfxButnQ4Y2lExmBysuaQzUQKBgQDmQ4tvP6wKHFwxljzG
ioyhJlWNsekJ306IVvLPd7fGfosJOIyVe/s1CJJa8hbNETZuW4TqyNVwBBKJYFHd
qBvJbOO5uxCb3kd9SoVR5GKM/XJm1344G1bBNil2fize+meza3YIs9bFP1DdJFeL
xxtIhfAAsehpsiaXLdO49/Cs/wKBgQDuAEc9DeIO5f6MuTijS2uvR8MrRcCap0PC
hYWjfeibYrg3jMAZuGmIEDY8UMwDIyTC5adsgnvbBwa9kuu1QJoOQkgpynSncdHq
rWRui1Vreqzwjwgo+vlPjP7UCNmmmsX5h4Lq/piKZRoNDHJuYbrsP4GLHYqXu708
nYriQn0THwKBgD+/s3q5RggGkf+oeGr/MK/szQLNnpb9G+DjMZX+rwJlRljnLwSz
rnigRuSm4Wkkus0SyVm3AGBiwD8IS6HySL8bRnDFOXPtth0Z5I8ni6E9Cc8vGeKu
rvYO6KvFgua4rL0foy4FnBA/10NiLFqmZQdPDYMjvhSFHen85gDvo9JBAoGAErlq
4V7KPinSEPgxgIdafXvD2w+ty30FU7R1gxtQxYjKvXQRWPStOMstuSkSI90L81lm
Bp9tZxnyac+5Q5VFiYGlc8Q90T6fKTti/19u5PnNyU83cSHto1iQY9YeaYEbujnz
DtHg5mJ13jjwd8awkg2c/3jfgnh+ttoeJxbcCDkCgYEA1wWLpUIeRchZHOqbYFoM
QEDRqvnHYqrd0cp9hTp+uyoCz9e8C8n54wKzgWEtkxsAfO3Pami4T+z+NiYeFpWn
+1SJIbJWbPf2elxD60UjEa8OeelSGiu5X5KlfPvyuVgX4qs9vUYQQiTHDZbzmDlV
I1Ru6xykeZLm6JI5YG+U0zg=
-----END PRIVATE KEY-----
";

    /// The token the mock endpoint mints.
    const MINTED: &str = "ya29.minted-token";

    /// Builds a key whose `token_uri` points at `server`.
    fn source(server: &MockServer) -> TokenSource {
        let document = json!({
            "type": "service_account",
            "private_key": SERVICE_ACCOUNT_PEM,
            "client_email": "symphony@demo-project.iam.gserviceaccount.com",
            "token_uri": format!("{}/token", server.uri()),
        });
        let key = ServiceAccountKey::from_json(&document, "<test>").expect("parse the key");
        TokenSource::new(key)
    }

    /// A client with the default policy, which does not retry.
    fn client() -> HttpClient {
        HttpClient::new(HttpClientConfig::default()).expect("build a client")
    }

    /// Mounts a 200 `token` reply carrying [`MINTED`] and `expires_in` seconds.
    async fn mount_mint(server: &MockServer, expires_in: u64) {
        Mock::given(method("POST"))
            .and(path("/token"))
            .and(header("content-type", "application/x-www-form-urlencoded"))
            .and(body_string_contains(
                "grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": MINTED,
                "expires_in": expires_in,
                "token_type": "Bearer",
            })))
            .expect(1)
            .mount(server)
            .await;
    }

    #[tokio::test]
    async fn a_successful_exchange_sends_the_form_and_returns_the_token() {
        let server = MockServer::start().await;
        mount_mint(&server, 3600).await;
        let source = source(&server);

        let token = source
            .access_token(&client(), Duration::from_secs(10))
            .await
            .expect("mint a token");
        assert_eq!(token.expose_secret(), MINTED);

        let requests = server.received_requests().await.expect("recorded requests");
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].url.path(), "/token");
        let body = String::from_utf8(requests[0].body.clone()).expect("a utf-8 form body");
        let assertion = body
            .split('&')
            .find_map(|pair| pair.strip_prefix("assertion="))
            .expect("the form carries the assertion");
        let segments: Vec<&str> = assertion.split('.').collect();
        assert_eq!(segments.len(), 3);
        let header: Value = serde_json::from_slice(
            &base64::engine::general_purpose::URL_SAFE_NO_PAD
                .decode(segments[0])
                .expect("decode the header"),
        )
        .expect("header JSON");
        assert_eq!(header["alg"], "RS256");
        let claims: Value = serde_json::from_slice(
            &base64::engine::general_purpose::URL_SAFE_NO_PAD
                .decode(segments[1])
                .expect("decode the claims"),
        )
        .expect("claims JSON");
        assert_eq!(
            claims["iss"],
            "symphony@demo-project.iam.gserviceaccount.com"
        );
        assert_eq!(claims["aud"], format!("{}/token", server.uri()));
    }

    #[tokio::test]
    async fn a_fresh_token_is_reused_without_a_second_request() {
        let server = MockServer::start().await;
        mount_mint(&server, 3600).await;
        let source = source(&server);
        let client = client();

        let first = source
            .access_token(&client, Duration::from_secs(10))
            .await
            .expect("mint");
        let second = source
            .access_token(&client, Duration::from_secs(10))
            .await
            .expect("reuse");
        assert_eq!(first.expose_secret(), second.expose_secret());
        assert_eq!(
            server
                .received_requests()
                .await
                .expect("recorded requests")
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn a_token_expiring_inside_the_margin_is_minted_again() {
        for expires_in in [0, 30] {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/token"))
                .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                    "access_token": MINTED,
                    "expires_in": expires_in,
                })))
                .mount(&server)
                .await;
            let source = source(&server);
            let client = client();

            for _round in 0..3 {
                let _ = source
                    .access_token(&client, Duration::from_secs(10))
                    .await
                    .expect("mint");
            }
            assert_eq!(
                server
                    .received_requests()
                    .await
                    .expect("recorded requests")
                    .len(),
                3,
                "expires_in {expires_in} must not be cached"
            );
        }
    }

    #[tokio::test]
    async fn concurrent_callers_share_one_mint() {
        let server = MockServer::start().await;
        mount_mint(&server, 3600).await;
        let source = source(&server);
        let client = client();

        let (left, right) = tokio::join!(
            source.access_token(&client, Duration::from_secs(10)),
            source.access_token(&client, Duration::from_secs(10))
        );
        assert_eq!(
            left.expect("first mint").expose_secret(),
            right.expect("second mint").expose_secret()
        );
        assert_eq!(
            server
                .received_requests()
                .await
                .expect("recorded requests")
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn an_expires_in_string_is_accepted() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "access_token": MINTED,
                "expires_in": "3600",
            })))
            .mount(&server)
            .await;
        let source = source(&server);
        let client = client();

        for _round in 0..2 {
            let token = source
                .access_token(&client, Duration::from_secs(10))
                .await
                .expect("mint");
            assert_eq!(token.expose_secret(), MINTED);
        }
        assert_eq!(
            server
                .received_requests()
                .await
                .expect("recorded requests")
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn a_rejected_exchange_names_the_endpoint_and_the_oauth_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/token"))
            .respond_with(ResponseTemplate::new(401).set_body_json(json!({
                "error": "invalid_grant",
                "error_description": "Invalid JWT",
            })))
            .mount(&server)
            .await;
        let source = source(&server);

        let error = source
            .access_token(&client(), Duration::from_secs(10))
            .await
            .expect_err("a 401 must fail");
        assert_eq!(error.kind(), crate::error::ErrorKind::External);
        let message = error.to_string();
        assert!(message.contains("401"), "{message}");
        assert!(message.contains("/token"), "{message}");
        assert!(message.contains("invalid_grant"), "{message}");
        assert!(message.contains("Invalid JWT"), "{message}");
        assert!(!message.contains("eyJ"), "{message}");
        assert!(!message.contains("assertion"), "{message}");
    }

    #[tokio::test]
    async fn a_server_error_with_an_empty_body_is_reported() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/token"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;
        let source = source(&server);

        let error = source
            .access_token(&client(), Duration::from_secs(10))
            .await
            .expect_err("a 500 must fail");
        assert_eq!(error.kind(), crate::error::ErrorKind::External);
        let message = error.to_string();
        assert!(message.contains("500"), "{message}");
        assert!(message.contains("<empty body>"), "{message}");
        assert!(!message.contains("eyJ"), "{message}");
    }

    #[tokio::test]
    async fn an_unusable_reply_is_an_error() {
        for body in [
            json!({ "token_type": "Bearer" }),
            json!({ "access_token": "   " }),
            json!({ "access_token": 7 }),
        ] {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/token"))
                .respond_with(ResponseTemplate::new(200).set_body_json(body.clone()))
                .mount(&server)
                .await;
            let source = source(&server);
            let error = source
                .access_token(&client(), Duration::from_secs(10))
                .await
                .expect_err("an unusable reply must fail");
            assert_eq!(error.kind(), crate::error::ErrorKind::External, "{body}");
            assert!(error.to_string().contains("/token"), "{body}: {error}");
        }
    }

    #[tokio::test]
    async fn a_reply_that_is_not_json_is_an_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/token"))
            .respond_with(ResponseTemplate::new(200).set_body_string("<html>nope</html>"))
            .mount(&server)
            .await;
        let source = source(&server);
        let error = source
            .access_token(&client(), Duration::from_secs(10))
            .await
            .expect_err("a non-JSON reply must fail");
        assert_eq!(error.kind(), crate::error::ErrorKind::External);
        assert!(error.to_string().contains("/token"), "{error}");
    }

    #[test]
    fn debug_reports_the_endpoint_and_never_the_token() {
        let source = TokenSource::new(
            ServiceAccountKey::from_json(
                &json!({
                    "type": "service_account",
                    "private_key": SERVICE_ACCOUNT_PEM,
                    "client_email": "symphony@example.com",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }),
                "<test>",
            )
            .expect("parse"),
        );
        let rendered = format!("{source:?}");
        assert!(
            rendered.contains("https://oauth2.googleapis.com/token"),
            "{rendered}"
        );
        assert!(rendered.contains("cached"), "{rendered}");
        assert!(!rendered.contains(MINTED), "{rendered}");
        assert!(!rendered.contains("MII"), "{rendered}");
    }

    #[test]
    fn the_exchange_failure_renders_the_documented_shape() {
        let failure = TokenExchangeFailure::new(
            "https://oauth2.googleapis.com/token",
            400,
            br#"{"error":{"message":"bad assertion"},"error_description":"nope"}"#,
            "",
        );
        let message = failure.to_string();
        assert_eq!(
            message,
            "service-account token exchange failed with HTTP 400 for \
             https://oauth2.googleapis.com/token: bad assertion: nope"
        );
        let failure = TokenExchangeFailure::new("https://t", 502, b"upstream \"down\"", "");
        assert!(
            failure.to_string().contains("upstream \"down\""),
            "{failure}"
        );
    }

    #[test]
    fn an_echoed_assertion_is_scrubbed_from_the_failure() {
        const ASSERTION: &str = "eyJhbGciOiJSUzI1NiJ9.eyJpc3MiOiJ4In0.c2ln";
        let body =
            format!(r#"{{"error":"invalid_grant","error_description":"rejected {ASSERTION}"}}"#);
        let failure = TokenExchangeFailure::new("https://t", 400, body.as_bytes(), ASSERTION);
        let message = failure.to_string();
        assert!(!message.contains(ASSERTION), "{message}");
        assert!(!message.contains("c2ln"), "{message}");
        assert!(message.contains("<redacted-assertion>"), "{message}");
        assert!(message.contains("invalid_grant"), "{message}");

        let echoed_form = percent_encode(ASSERTION);
        let body =
            format!("{{\"error\":\"invalid_grant\",\"error_description\":\"{echoed_form}\"}}");
        let failure = TokenExchangeFailure::new("https://t", 400, body.as_bytes(), ASSERTION);
        assert!(!failure.to_string().contains(&echoed_form), "{failure}");
    }

    #[test]
    fn form_values_are_percent_encoded() {
        assert_eq!(percent_encode("abcXYZ0-_.~"), "abcXYZ0-_.~");
        assert_eq!(percent_encode("a:b/c+d"), "a%3Ab%2Fc%2Bd");
        assert_eq!(percent_encode("é"), "%C3%A9");
    }
}
