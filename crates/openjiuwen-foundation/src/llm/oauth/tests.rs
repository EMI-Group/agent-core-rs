//! Transport tests for the `OpenAI` account OAuth client and its credential store.
//!
//! Every HTTP test talks to a [`MockServer`] on loopback and every file test uses
//! a [`tempfile`] directory, so the suite needs no network, no credential, and
//! never touches `~/.openjiuwen`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Duration;

use chrono::{TimeDelta, Utc};
use openjiuwen_common::config::env;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_common::security::REDACTED;
use serde_json::{json, Value};
use wiremock::matchers::{body_json, body_string, body_string_contains, header, method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

use super::client::OpenAiOAuthClient;
use super::store::{
    DeviceCode, FileTokenStore, InMemoryTokenStore, OAuthToken, TokenStore, AUTH_FILE_ENV,
    AUTH_MODE, DEFAULT_BASE_URL, REFRESH_SKEW,
};
use super::support::{expiry_in, jwt, oauth_client, secret, CLIENT_ID};

/// The redirect URI the device flow reports to the token endpoint.
const REDIRECT_URI: &str = "https://auth.openai.com/deviceauth/callback";

/// Encodes a form body the same way the client does, for exact assertions.
fn form(fields: &[(&str, &str)]) -> String {
    let mut serializer = url::form_urlencoded::Serializer::new(String::new());
    for (key, value) in fields {
        serializer.append_pair(key, value);
    }
    serializer.finish()
}

// --- device authorization ---------------------------------------------------

#[tokio::test]
async fn request_device_code_posts_the_client_id_and_parses_the_response() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/accounts/deviceauth/usercode"))
        .and(header("content-type", "application/json"))
        .and(body_json(json!({ "client_id": CLIENT_ID })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "user_code": "ABCD-EFGH",
            "device_auth_id": "device-1",
            "interval": 3,
            "expires_in": 600
        })))
        .expect(1)
        .mount(&server)
        .await;

    let device = oauth_client(&server)
        .request_device_code()
        .await
        .expect("the device code");

    assert_eq!(device.user_code, "ABCD-EFGH");
    assert_eq!(device.device_auth_id, "device-1");
    assert_eq!(device.interval, 3);
    assert_eq!(device.expires_in, 600);
}

#[tokio::test]
async fn request_device_code_raises_a_short_advertised_interval_to_three() {
    // ports `max(3, interval)` from request_openai_account_device_code.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/accounts/deviceauth/usercode"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "user_code": "ABCD-EFGH",
            "device_auth_id": "device-1",
            "interval": 1,
            "expires_in": 600
        })))
        .mount(&server)
        .await;

    let device = oauth_client(&server)
        .request_device_code()
        .await
        .expect("the device code");

    assert_eq!(
        device.interval, 3,
        "the endpoint is never polled faster than every three seconds"
    );
}

#[tokio::test]
async fn request_device_code_redacts_a_rejected_response() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/accounts/deviceauth/usercode"))
        .respond_with(ResponseTemplate::new(500).set_body_string(r#"{"error":"sk-live-secret"}"#))
        .mount(&server)
        .await;

    let error = oauth_client(&server)
        .request_device_code()
        .await
        .expect_err("the provider rejects the request");

    assert_eq!(error.kind(), ErrorKind::Llm);
    let message = error.to_string();
    assert!(message.contains("HTTP 500"), "message: {message}");
    assert!(message.contains(REDACTED), "message: {message}");
    assert!(!message.contains("sk-live-secret"), "message: {message}");
}

#[tokio::test]
async fn poll_device_token_waits_through_pending_and_exchanges_the_code() {
    let server = MockServer::start().await;
    let attempts = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&attempts);
    Mock::given(method("POST"))
        .and(path("/api/accounts/deviceauth/token"))
        .and(body_json(json!({
            "device_auth_id": "device-1",
            "user_code": "ABCD-EFGH"
        })))
        .respond_with(
            move |_request: &Request| match counter.fetch_add(1, Ordering::SeqCst) {
                0 => ResponseTemplate::new(403),
                1 => ResponseTemplate::new(404),
                _ => ResponseTemplate::new(200).set_body_json(json!({
                    "authorization_code": "auth-code",
                    "code_verifier": "verifier-1"
                })),
            },
        )
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .and(header("content-type", "application/x-www-form-urlencoded"))
        .and(body_string_contains("grant_type=authorization_code"))
        .and(body_string_contains("code=auth-code"))
        .and(body_string_contains("code_verifier=verifier-1"))
        .and(body_string_contains(format!("client_id={CLIENT_ID}")))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "access_token": "access-1",
            "refresh_token": "refresh-1",
            "expires_in": 3600,
            "token_type": "Bearer"
        })))
        .expect(1)
        .mount(&server)
        .await;

    let code = DeviceCode::new("ABCD-EFGH", "device-1")
        .with_interval(0)
        .with_expires_in(120);
    let token = oauth_client(&server)
        .poll_device_token_with_deadline(&code, Duration::from_secs(4))
        .await
        .expect("the tokens");

    assert_eq!(token.access_token.expose_secret(), "access-1");
    assert_eq!(secret(token.refresh_token.as_ref()), Some("refresh-1"));
    assert!(token.expires_at.expect("expiry") > Utc::now());
    assert!(
        attempts.load(Ordering::SeqCst) >= 3,
        "expected the two pending polls and the final one"
    );
}

#[tokio::test]
async fn poll_device_token_returns_a_token_carried_by_the_poll_response() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/accounts/deviceauth/token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "access_token": "direct-access",
            "refresh_token": "direct-refresh",
            "expires_in": 60
        })))
        .mount(&server)
        .await;

    let code = DeviceCode::new("ABCD-EFGH", "device-1")
        .with_interval(0)
        .with_expires_in(30);
    let token = oauth_client(&server)
        .poll_device_token(&code)
        .await
        .expect("the tokens");

    assert_eq!(token.access_token.expose_secret(), "direct-access");
    assert_eq!(secret(token.refresh_token.as_ref()), Some("direct-refresh"));
}

#[tokio::test]
async fn poll_device_token_reports_an_expired_device_code() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/accounts/deviceauth/token"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;

    let code = DeviceCode::new("ABCD-EFGH", "device-1")
        .with_interval(0)
        .with_expires_in(1);
    let error = oauth_client(&server)
        .poll_device_token(&code)
        .await
        .expect_err("the device code expires");

    assert_eq!(error.kind(), ErrorKind::Llm);
    let message = error.to_string();
    assert!(message.contains("expired"), "message: {message}");
    assert!(
        message.contains("device login"),
        "the message tells the caller to start over: {message}"
    );
}

// --- token exchange and refresh ---------------------------------------------

#[tokio::test]
async fn exchange_code_posts_the_authorization_grant_form() {
    let server = MockServer::start().await;
    let expected = form(&[
        ("grant_type", "authorization_code"),
        ("code", "auth-code"),
        ("redirect_uri", REDIRECT_URI),
        ("client_id", CLIENT_ID),
        ("code_verifier", "verifier-1"),
    ]);
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .and(header("content-type", "application/x-www-form-urlencoded"))
        .and(body_string(expected))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "access_token": "access-1",
            "refresh_token": "refresh-1",
            "id_token": "id-1",
            "expires_in": 3600,
            "token_type": "Bearer",
            "scope": "openid profile"
        })))
        .expect(1)
        .mount(&server)
        .await;

    let token = oauth_client(&server)
        .exchange_code("auth-code", "verifier-1", REDIRECT_URI)
        .await
        .expect("the tokens");

    assert_eq!(token.access_token.expose_secret(), "access-1");
    assert_eq!(secret(token.refresh_token.as_ref()), Some("refresh-1"));
    assert_eq!(secret(token.id_token.as_ref()), Some("id-1"));
    assert_eq!(token.token_type, "Bearer");
    assert_eq!(token.scope.as_deref(), Some("openid profile"));
    let remaining = token
        .expires_at
        .expect("the token carries an expiry")
        .signed_duration_since(Utc::now());
    assert!(
        remaining > TimeDelta::try_seconds(3_000).expect("a duration"),
        "remaining: {remaining}"
    );
    assert!(remaining <= TimeDelta::try_seconds(3_600).expect("a duration"));
}

#[tokio::test]
async fn exchange_code_reports_rate_limiting_without_relogin() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(ResponseTemplate::new(429).set_body_string("slow down"))
        .mount(&server)
        .await;

    let error = oauth_client(&server)
        .exchange_code("auth-code", "verifier-1", REDIRECT_URI)
        .await
        .expect_err("the endpoint is rate limited");

    assert_eq!(error.kind(), ErrorKind::Llm);
    let message = error.to_string();
    assert!(message.contains("rate limited"), "message: {message}");
    assert!(
        message.contains("still valid"),
        "rate limiting must not read as a credential failure: {message}"
    );
}

#[tokio::test]
async fn refresh_rotates_the_refresh_token() {
    let server = MockServer::start().await;
    let expected = form(&[
        ("grant_type", "refresh_token"),
        ("refresh_token", "old-refresh"),
        ("client_id", CLIENT_ID),
    ]);
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .and(header("content-type", "application/x-www-form-urlencoded"))
        .and(body_string(expected))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "access_token": "rotated-access",
            "refresh_token": "rotated-refresh",
            "expires_in": 3600
        })))
        .expect(1)
        .mount(&server)
        .await;

    let previous = OAuthToken::new("old-access")
        .with_refresh_token("old-refresh")
        .with_expires_at(expiry_in(-10));
    let refreshed = oauth_client(&server)
        .refresh(&previous)
        .await
        .expect("the refreshed token");

    assert_eq!(refreshed.access_token.expose_secret(), "rotated-access");
    assert_eq!(
        secret(refreshed.refresh_token.as_ref()),
        Some("rotated-refresh")
    );
    assert!(refreshed.expires_at.expect("expiry") > Utc::now());

    // The grant form carries exactly three parameters: no `scope`, no
    // `redirect_uri` (Python parity: refresh_openai_account_oauth).
    let requests = server
        .received_requests()
        .await
        .expect("the recorded requests");
    let body = String::from_utf8_lossy(&requests.first().expect("the refresh request").body);
    assert!(!body.contains("scope="), "body: {body}");
    assert!(!body.contains("redirect_uri="), "body: {body}");
    assert_eq!(body.matches('=').count(), 3, "body: {body}");
}

#[tokio::test]
async fn refresh_keeps_the_previous_refresh_token_when_the_response_omits_it() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "access_token": "new-access",
            "expires_in": 3600
        })))
        .mount(&server)
        .await;

    let previous = OAuthToken::new("old-access")
        .with_refresh_token("old-refresh")
        .with_id_token("old-id")
        .with_scope("openid")
        .with_expires_at(expiry_in(-10));
    let refreshed = oauth_client(&server)
        .refresh(&previous)
        .await
        .expect("the refreshed token");

    assert_eq!(refreshed.access_token.expose_secret(), "new-access");
    assert_eq!(
        secret(refreshed.refresh_token.as_ref()),
        Some("old-refresh")
    );
    assert_eq!(secret(refreshed.id_token.as_ref()), Some("old-id"));
    assert_eq!(refreshed.scope.as_deref(), Some("openid"));
    assert!(refreshed.expires_at.expect("expiry") > Utc::now());
}

// --- access_token -----------------------------------------------------------

#[tokio::test]
async fn access_token_refreshes_a_token_inside_the_skew_and_saves_it() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .and(body_string_contains("grant_type=refresh_token"))
        .and(body_string_contains("refresh_token=old-refresh"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "access_token": "fresh-access",
            "refresh_token": "fresh-refresh",
            "expires_in": 3600
        })))
        .expect(1)
        .mount(&server)
        .await;

    let store = InMemoryTokenStore::new();
    let expiring = OAuthToken::new("stale-access")
        .with_refresh_token("old-refresh")
        .with_expires_at(expiry_in(60));
    assert!(
        expiring.is_expired(REFRESH_SKEW),
        "a token expiring in 60 s is inside the 120 s skew"
    );
    store.save(&expiring).await.expect("the stored token");

    let token = oauth_client(&server)
        .access_token(&store, false)
        .await
        .expect("the usable token");

    assert_eq!(token.access_token.expose_secret(), "fresh-access");
    assert_eq!(secret(token.refresh_token.as_ref()), Some("fresh-refresh"));

    let stored = store.load().await.expect("the reload").expect("the token");
    assert_eq!(stored.access_token.expose_secret(), "fresh-access");
    assert_eq!(secret(stored.refresh_token.as_ref()), Some("fresh-refresh"));
}

#[tokio::test]
async fn access_token_force_refresh_ignores_the_remaining_lifetime() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "access_token": "forced-access",
            "refresh_token": "forced-refresh",
            "expires_in": 3600
        })))
        .expect(1)
        .mount(&server)
        .await;

    let store = InMemoryTokenStore::new();
    store
        .save(
            &OAuthToken::new("valid-access")
                .with_refresh_token("valid-refresh")
                .with_expires_at(expiry_in(3600)),
        )
        .await
        .expect("the stored token");

    let token = oauth_client(&server)
        .access_token(&store, true)
        .await
        .expect("the refreshed token");

    assert_eq!(token.access_token.expose_secret(), "forced-access");
    assert_eq!(
        store
            .load()
            .await
            .expect("the reload")
            .expect("the token")
            .access_token
            .expose_secret(),
        "forced-access"
    );
}

#[tokio::test]
async fn access_token_returns_a_valid_token_without_network_access() {
    let server = MockServer::start().await;
    let store = InMemoryTokenStore::new();
    store
        .save(&OAuthToken::new("access-1").with_expires_at(expiry_in(3600)))
        .await
        .expect("the stored token");

    let token = oauth_client(&server)
        .access_token(&store, false)
        .await
        .expect("the usable token");

    assert_eq!(token.access_token.expose_secret(), "access-1");
    assert!(
        server
            .received_requests()
            .await
            .expect("the recorded requests")
            .is_empty(),
        "a valid token must not trigger a refresh"
    );
}

#[tokio::test]
async fn access_token_without_a_stored_token_is_a_config_error() {
    let client = OpenAiOAuthClient::new().expect("the test client builds");
    let store = InMemoryTokenStore::new();

    let error = client
        .access_token(&store, false)
        .await
        .expect_err("nothing is stored");

    assert_eq!(error.kind(), ErrorKind::Config);
    let message = error.to_string();
    assert!(
        message.contains("device authorization flow"),
        "the message tells the caller what to run: {message}"
    );
}

#[tokio::test]
async fn access_token_without_a_refresh_token_reports_relogin() {
    let client = OpenAiOAuthClient::new().expect("the test client builds");
    let store = InMemoryTokenStore::new();
    store
        .save(&OAuthToken::new("stale-access").with_expires_at(expiry_in(-5)))
        .await
        .expect("the stored token");

    let error = client
        .access_token(&store, false)
        .await
        .expect_err("the token cannot be refreshed");

    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(
        error.to_string().contains("refresh token"),
        "message: {error}"
    );
}

#[tokio::test]
async fn access_token_never_refreshes_a_token_without_a_resolvable_expiry() {
    let server = MockServer::start().await;
    let store = InMemoryTokenStore::new();
    // A token whose expiry nothing can resolve: no recorded `expires_at`, and an
    // access token that is not a JWT.
    store
        .save(&OAuthToken::new("opaque-access").with_refresh_token("old-refresh"))
        .await
        .expect("the stored token");

    let token = oauth_client(&server)
        .access_token(&store, false)
        .await
        .expect("the usable token");

    assert_eq!(token.access_token.expose_secret(), "opaque-access");
    assert!(
        server
            .received_requests()
            .await
            .expect("the recorded requests")
            .is_empty(),
        "an unresolvable expiry never triggers a refresh"
    );
}

#[tokio::test]
async fn access_token_refreshes_a_token_whose_jwt_claim_has_passed() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .and(body_string_contains("refresh_token=old-refresh"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "access_token": "fresh-access",
            "expires_in": 3600
        })))
        .expect(1)
        .mount(&server)
        .await;

    let expired_claim = Utc::now().timestamp() - 60;
    let store = InMemoryTokenStore::new();
    store
        .save(
            &OAuthToken::new(jwt(json!({ "exp": expired_claim })))
                .with_refresh_token("old-refresh"),
        )
        .await
        .expect("the stored token");

    let token = oauth_client(&server)
        .access_token(&store, false)
        .await
        .expect("the refreshed token");

    assert_eq!(token.access_token.expose_secret(), "fresh-access");
}

// --- file-backed store ------------------------------------------------------

#[tokio::test]
async fn file_store_round_trips_and_preserves_other_providers() {
    let directory = tempfile::tempdir().expect("a temp directory");
    let path = directory.path().join("nested").join("auth.json");
    std::fs::create_dir_all(path.parent().expect("the parent directory"))
        .expect("the parent directory is created");
    std::fs::write(
        &path,
        serde_json::to_vec(&json!({
            "providers": { "Other": { "tokens": { "access_token": "other" } } },
            "telemetry": true
        }))
        .expect("the seeded document"),
    )
    .expect("the seeded store");

    let store = FileTokenStore::new(&path);
    assert_eq!(store.path(), path.as_path());
    assert_eq!(store.base_url(), DEFAULT_BASE_URL);
    assert!(store.load().await.expect("the load").is_none());

    let token = OAuthToken::new("access-1")
        .with_refresh_token("refresh-1")
        .with_scope("openid")
        .with_expires_at(expiry_in(3600));
    store.save(&token).await.expect("the saved token");

    let document: Value =
        serde_json::from_slice(&std::fs::read(&path).expect("the store is readable"))
            .expect("the store is JSON");
    let entry = &document["providers"]["OpenAIAccount"];
    assert_eq!(entry["auth_mode"], AUTH_MODE);
    assert_eq!(entry["base_url"], DEFAULT_BASE_URL);
    assert_eq!(entry["tokens"]["access_token"], "access-1");
    assert_eq!(entry["tokens"]["token_type"], "Bearer");
    assert_eq!(
        document["providers"]["Other"]["tokens"]["access_token"],
        "other"
    );
    assert_eq!(document["telemetry"], true);

    let loaded = store.load().await.expect("the load").expect("the token");
    assert_eq!(loaded.access_token.expose_secret(), "access-1");
    assert_eq!(secret(loaded.refresh_token.as_ref()), Some("refresh-1"));
    assert_eq!(loaded.scope.as_deref(), Some("openid"));
    assert_eq!(loaded.expires_at, token.expires_at);
}

#[tokio::test]
async fn file_store_clear_removes_only_the_openai_account_entry() {
    let directory = tempfile::tempdir().expect("a temp directory");
    let path = directory.path().join("auth.json");
    let store = FileTokenStore::new(&path);
    store
        .save(&OAuthToken::new("access-1").with_refresh_token("refresh-1"))
        .await
        .expect("the saved token");

    let mut document: Value =
        serde_json::from_slice(&std::fs::read(&path).expect("the store is readable"))
            .expect("the store is JSON");
    document["providers"]["Other"] = json!({ "tokens": { "access_token": "other" } });
    std::fs::write(&path, serde_json::to_vec(&document).expect("the document"))
        .expect("the rewritten store");

    store.clear().await.expect("the cleared token");

    assert!(store.load().await.expect("the load").is_none());
    let document: Value =
        serde_json::from_slice(&std::fs::read(&path).expect("the store is readable"))
            .expect("the store is JSON");
    assert!(document["providers"].get("OpenAIAccount").is_none());
    assert_eq!(
        document["providers"]["Other"]["tokens"]["access_token"],
        "other"
    );

    // Clearing an already-clear store is a no-op, not an error.
    store.clear().await.expect("the second clear");
}

#[cfg(unix)]
#[tokio::test]
async fn file_store_writes_owner_only_permissions_and_leaves_no_temporary_file() {
    use std::os::unix::fs::PermissionsExt;

    let directory = tempfile::tempdir().expect("a temp directory");
    let path = directory.path().join("auth.json");
    let store = FileTokenStore::new(&path);

    store
        .save(&OAuthToken::new("access-1"))
        .await
        .expect("the saved token");

    let mode = std::fs::metadata(&path)
        .expect("the store metadata")
        .permissions()
        .mode();
    assert_eq!(mode & 0o777, 0o600, "the credential file is owner-only");

    let leftovers: Vec<PathBuf> = std::fs::read_dir(directory.path())
        .expect("the directory listing")
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.to_string_lossy().ends_with(".tmp"))
        .collect();
    assert!(leftovers.is_empty(), "temporary files: {leftovers:?}");
}

#[tokio::test]
async fn file_store_reports_an_unreadable_store() {
    let directory = tempfile::tempdir().expect("a temp directory");
    let path = directory.path().join("auth.json");
    std::fs::write(&path, b"not json").expect("the corrupted store");

    let store = FileTokenStore::new(&path);
    let error = store.load().await.expect_err("the store is not JSON");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.to_string().contains("not valid JSON"),
        "message: {error}"
    );
}

// --- path resolution --------------------------------------------------------

/// Serializes tests that mutate the process environment.
fn env_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Every variable [`FileTokenStore::default_path`] may read.
const ENV_VARS: &[&str] = &[AUTH_FILE_ENV, env::OPENJIUWEN_HOME, "HOME", "USERPROFILE"];

/// Clears every variable the path resolver reads and restores it on drop.
struct EnvGuard {
    _lock: MutexGuard<'static, ()>,
    saved: Vec<(&'static str, Option<String>)>,
}

impl EnvGuard {
    fn clear() -> Self {
        let lock = env_lock();
        let saved: Vec<(&'static str, Option<String>)> = ENV_VARS
            .iter()
            .map(|name| (*name, std::env::var(name).ok()))
            .collect();
        for name in ENV_VARS {
            std::env::remove_var(name);
        }
        Self { _lock: lock, saved }
    }

    #[allow(clippy::unused_self)]
    fn set(&self, name: &str, value: &str) {
        std::env::set_var(name, value);
    }

    #[allow(clippy::unused_self)]
    fn remove(&self, name: &str) {
        std::env::remove_var(name);
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (name, value) in &self.saved {
            match value {
                Some(value) => std::env::set_var(name, value),
                None => std::env::remove_var(name),
            }
        }
    }
}

#[test]
fn default_path_follows_the_environment() {
    let env = EnvGuard::clear();

    env.set(AUTH_FILE_ENV, "/tmp/openjiuwen-oauth-test/auth.json");
    assert_eq!(
        FileTokenStore::default_path().expect("the auth file wins"),
        PathBuf::from("/tmp/openjiuwen-oauth-test/auth.json")
    );

    env.remove(AUTH_FILE_ENV);
    env.set(env::OPENJIUWEN_HOME, "/home/tester/runtime");
    assert_eq!(
        FileTokenStore::default_path().expect("the home override"),
        PathBuf::from("/home/tester/runtime/auth.json")
    );

    env.remove(env::OPENJIUWEN_HOME);
    env.set("HOME", "/home/tester");
    assert_eq!(
        FileTokenStore::default_path().expect("the home directory"),
        PathBuf::from("/home/tester/.openjiuwen/auth.json")
    );

    env.remove("HOME");
    env.remove("USERPROFILE");
    let error = FileTokenStore::default_path().expect_err("nothing resolves");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(
        error.to_string().contains(AUTH_FILE_ENV),
        "the message names the override: {error}"
    );
}
