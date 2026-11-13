//! Shared helpers for the `oauth` test trees.
//!
//! The transport, refresh, and token suites all drive a mock server and need the
//! same expiry arithmetic and JWT builder, so those helpers live here instead of
//! being copied into each test module.

use chrono::{DateTime, Utc};
use openjiuwen_common::security::SecretString;
use serde_json::{json, Value};
use wiremock::MockServer;

use super::client::OpenAiOAuthClient;
use super::store::OAuthToken;

/// Client id every test configures: a literal proves the configured value wins
/// over the built-in default.
pub(super) const CLIENT_ID: &str = "test-client-id";

/// Builds a client pointed at the mock server's root.
pub(super) fn oauth_client(server: &MockServer) -> OpenAiOAuthClient {
    OpenAiOAuthClient::new()
        .expect("the test client builds")
        .with_base_url(server.uri())
        .with_client_id(CLIENT_ID)
}

/// Returns an expiry `seconds` from now, truncated to whole seconds.
pub(super) fn expiry_in(seconds: i64) -> DateTime<Utc> {
    DateTime::from_timestamp(Utc::now().timestamp() + seconds, 0).expect("an in-range timestamp")
}

/// Returns the fixed expiry the JWT tests use.
pub(super) fn jwt_expiry() -> DateTime<Utc> {
    DateTime::from_timestamp(1_900_000_000, 0).expect("an in-range timestamp")
}

/// Builds an unsigned JWT whose payload is `claims`.
///
/// The signature is a placeholder: the client never verifies it, it only reads
/// the `exp` claim.
pub(super) fn jwt(claims: Value) -> String {
    use base64::Engine as _;

    let encode = |value: &Value| {
        base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::to_vec(value).expect("the JWT part serializes"))
    };
    format!(
        "{}.{}.signature",
        encode(&json!({ "alg": "none", "typ": "JWT" })),
        encode(&claims)
    )
}

/// Reads a secret option back as plain text.
pub(super) fn secret(value: Option<&SecretString>) -> Option<&str> {
    value.map(SecretString::expose_secret)
}

/// Builds the token a refresh test replaces.
pub(super) fn refreshable() -> OAuthToken {
    OAuthToken::new("old-access").with_refresh_token("old-refresh")
}
