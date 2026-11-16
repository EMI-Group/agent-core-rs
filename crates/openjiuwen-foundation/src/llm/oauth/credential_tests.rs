//! Login, status, and logout suites for the `OpenAI` account credentials.
//!
//! The device flow runs against a [`MockServer`] on loopback and every file test
//! uses a [`tempfile`] directory, so the suite needs no network, no credential, and
//! never touches `~/.openjiuwen`.

use std::sync::Arc;

use chrono::Utc;
use openjiuwen_common::error::ErrorKind;
use serde_json::{json, Value};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::credentials::{login, logout, status, OAuthCredentials};
use super::store::{
    FileTokenStore, InMemoryTokenStore, OAuthToken, TokenStore, AUTH_MODE, DEFAULT_BASE_URL,
};
use super::support::{expiry_in, oauth_client};

/// Mounts the three requests of a complete device login.
async fn mount_login(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/api/accounts/deviceauth/usercode"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "user_code": "ABCD-EFGH",
            "device_auth_id": "device-1",
            "interval": 1,
            "expires_in": 120
        })))
        .expect(1)
        .mount(server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/accounts/deviceauth/token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "authorization_code": "auth-code",
            "code_verifier": "verifier-1"
        })))
        .expect(1)
        .mount(server)
        .await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .and(header("content-type", "application/x-www-form-urlencoded"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "access_token": "access-1",
            "refresh_token": "refresh-1",
            "id_token": "id-1",
            "expires_in": 3600,
            "token_type": "Bearer",
            "scope": "openid profile"
        })))
        .expect(1)
        .mount(server)
        .await;
}

/// Reads the auth-store document at `path`.
fn read_document(path: &std::path::Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).expect("the store is readable"))
        .expect("the store is JSON")
}

// --- login ------------------------------------------------------------------

#[tokio::test]
async fn login_runs_the_device_flow_and_persists_the_token() {
    let server = MockServer::start().await;
    mount_login(&server).await;

    let directory = tempfile::tempdir().expect("a temp directory");
    let auth_path = directory.path().join("auth.json");
    let credentials = OAuthCredentials::at_path(&auth_path)
        .expect("the credentials build")
        .with_client(oauth_client(&server));

    let info = login(&credentials).await.expect("the login");

    assert_eq!(
        info.verification_uri,
        format!("{}/codex/device", server.uri()),
        "the verification URL is derived from the issuer"
    );
    assert_eq!(info.user_code, "ABCD-EFGH");
    assert_eq!(info.expires_in, 120);
    assert_eq!(
        info.interval, 3,
        "the advertised interval is floored at 3 s"
    );
    assert!(info.token.has_refresh_token);
    assert!(info.token.has_id_token);
    assert_eq!(info.token.token_type, "Bearer");
    assert_eq!(info.token.scope.as_deref(), Some("openid profile"));
    assert!(
        info.token.expires_at.expect("the expiry is known") > Utc::now(),
        "the reported lifetime becomes an absolute expiry"
    );

    let document = read_document(&auth_path);
    let entry = &document["providers"]["OpenAIAccount"];
    assert_eq!(entry["tokens"]["access_token"], "access-1");
    assert_eq!(entry["tokens"]["refresh_token"], "refresh-1");
    assert_eq!(entry["tokens"]["id_token"], "id-1");
    assert_eq!(entry["auth_mode"], AUTH_MODE);
    assert_eq!(entry["base_url"], DEFAULT_BASE_URL);
    assert!(
        auth_path.with_file_name("auth.json.lock").exists(),
        "the login persisted under the cross-process lock"
    );

    // Nothing that reaches a log line carries a credential: the returned summary
    // and the credentials handle expose only presence flags and metadata.
    for rendered in [format!("{info:?}"), format!("{credentials:?}")] {
        for secret in ["access-1", "refresh-1", "id-1"] {
            assert!(
                !rendered.contains(secret),
                "the rendered value leaks {secret}: {rendered}"
            );
        }
    }
}

#[tokio::test]
async fn login_reports_a_rejected_device_code() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/accounts/deviceauth/usercode"))
        .respond_with(ResponseTemplate::new(500).set_body_string(r#"{"error":"secret"}"#))
        .mount(&server)
        .await;

    let directory = tempfile::tempdir().expect("a temp directory");
    let auth_path = directory.path().join("auth.json");
    let credentials = OAuthCredentials::at_path(&auth_path)
        .expect("the credentials build")
        .with_client(oauth_client(&server));

    let error = login(&credentials).await.expect_err("the issuer rejects");

    assert_eq!(error.kind(), ErrorKind::Llm);
    assert!(
        !auth_path.exists(),
        "a failed login must not create the credential file"
    );
}

// --- status -----------------------------------------------------------------

#[tokio::test]
async fn status_reports_an_absent_credential() {
    let directory = tempfile::tempdir().expect("a temp directory");
    let auth_path = directory.path().join("auth.json");
    let credentials = OAuthCredentials::at_path(&auth_path).expect("the credentials build");

    let status = status(&credentials).await.expect("the status");

    assert!(!status.present);
    assert!(!status.expired);
    assert!(!status.is_valid());
    assert!(!status.has_refresh_token);
    assert_eq!(status.expires_at, None);
    assert_eq!(status.auth_path.as_deref(), Some(auth_path.as_path()));
    assert_eq!(status.auth_mode, None);
    assert!(
        !auth_path.exists(),
        "reporting a status never creates the store"
    );
}

#[tokio::test]
async fn status_reports_a_valid_credential() {
    let directory = tempfile::tempdir().expect("a temp directory");
    let auth_path = directory.path().join("auth.json");
    let credentials = OAuthCredentials::at_path(&auth_path).expect("the credentials build");
    let token = OAuthToken::new("access-1")
        .with_refresh_token("refresh-1")
        .with_expires_at(expiry_in(3600));
    credentials
        .store()
        .save(&token)
        .await
        .expect("the saved token");

    let status = status(&credentials).await.expect("the status");

    assert!(status.present);
    assert!(!status.expired);
    assert!(status.is_valid());
    assert!(status.has_refresh_token);
    assert_eq!(status.expires_at, token.expires_at);
    assert_eq!(status.auth_mode.as_deref(), Some(AUTH_MODE));
    assert_eq!(status.auth_path.as_deref(), Some(auth_path.as_path()));
}

#[tokio::test]
async fn status_reports_an_expired_credential() {
    let directory = tempfile::tempdir().expect("a temp directory");
    let auth_path = directory.path().join("auth.json");
    let credentials = OAuthCredentials::at_path(&auth_path).expect("the credentials build");
    credentials
        .store()
        .save(&OAuthToken::new("stale-access").with_expires_at(expiry_in(-60)))
        .await
        .expect("the saved token");

    let status = status(&credentials).await.expect("the status");

    assert!(status.present);
    assert!(status.expired);
    assert!(!status.is_valid());
    assert!(!status.has_refresh_token);
    assert!(status.expires_at.expect("the expiry is known") < Utc::now());
}

#[tokio::test]
async fn status_reports_a_credential_without_a_resolvable_expiry() {
    let directory = tempfile::tempdir().expect("a temp directory");
    let auth_path = directory.path().join("auth.json");
    let credentials = OAuthCredentials::at_path(&auth_path).expect("the credentials build");
    // No expiry was recorded and the access token is not a JWT: nothing resolves.
    credentials
        .store()
        .save(&OAuthToken::new("opaque-access").with_refresh_token("refresh-1"))
        .await
        .expect("the saved token");

    let status = status(&credentials).await.expect("the status");

    assert!(status.present);
    assert_eq!(
        status.expires_at, None,
        "an unresolvable expiry stays unreported"
    );
    assert!(
        status.expired,
        "an unresolvable expiry counts as expired, so the next use refreshes"
    );
    assert!(!status.is_valid());
    assert!(status.has_refresh_token);
}

#[tokio::test]
async fn status_uses_the_configured_skew() {
    let directory = tempfile::tempdir().expect("a temp directory");
    let auth_path = directory.path().join("auth.json");
    let credentials = OAuthCredentials::at_path(&auth_path)
        .expect("the credentials build")
        .with_skew(std::time::Duration::from_secs(1));
    credentials
        .store()
        .save(&OAuthToken::new("access-1").with_expires_at(expiry_in(60)))
        .await
        .expect("the saved token");

    let status = status(&credentials).await.expect("the status");

    assert!(
        !status.expired,
        "60 s of validity is outside an explicitly small skew"
    );
}

#[tokio::test]
async fn status_reports_a_custom_store_without_a_path() {
    let store = Arc::new(InMemoryTokenStore::new());
    store
        .save(&OAuthToken::new("access-1").with_expires_at(expiry_in(3600)))
        .await
        .expect("the saved token");
    let credentials = OAuthCredentials::from_store(store).expect("the credentials build");

    let status = status(&credentials).await.expect("the status");

    assert!(status.present);
    assert!(status.is_valid());
    assert_eq!(status.auth_path, None, "an in-memory store has no file");
    assert_eq!(status.auth_mode, None);
}

#[tokio::test]
async fn status_reports_a_corrupt_store() {
    let directory = tempfile::tempdir().expect("a temp directory");
    let auth_path = directory.path().join("auth.json");
    std::fs::write(&auth_path, b"not json").expect("the corrupt store");
    let credentials = OAuthCredentials::at_path(&auth_path).expect("the credentials build");

    let error = status(&credentials)
        .await
        .expect_err("the store is not JSON");

    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.to_string().contains("not valid JSON"),
        "message: {error}"
    );
}

// --- logout -----------------------------------------------------------------

#[tokio::test]
async fn logout_removes_the_credential_once() {
    let directory = tempfile::tempdir().expect("a temp directory");
    let auth_path = directory.path().join("auth.json");
    let credentials = OAuthCredentials::at_path(&auth_path).expect("the credentials build");
    credentials
        .store()
        .save(&OAuthToken::new("access-1").with_refresh_token("refresh-1"))
        .await
        .expect("the saved token");

    assert!(
        logout(&credentials).await.expect("the first logout"),
        "a stored credential is removed"
    );
    assert!(
        credentials
            .store()
            .load()
            .await
            .expect("the store read")
            .is_none(),
        "the store reports no credential afterwards"
    );
    let document = read_document(&auth_path);
    assert!(document["providers"].get("OpenAIAccount").is_none());

    assert!(
        !logout(&credentials).await.expect("the second logout"),
        "logging out again is a no-op, not an error"
    );
}

#[tokio::test]
async fn logout_is_idempotent_without_a_store() {
    let directory = tempfile::tempdir().expect("a temp directory");
    let auth_path = directory.path().join("absent").join("auth.json");
    let credentials = OAuthCredentials::at_path(&auth_path).expect("the credentials build");

    assert!(!logout(&credentials).await.expect("the logout"));
    assert!(!auth_path.exists(), "nothing is created just to log out");
}

#[tokio::test]
async fn logout_clears_a_custom_store() {
    let store = Arc::new(InMemoryTokenStore::new());
    store
        .save(&OAuthToken::new("access-1"))
        .await
        .expect("the saved token");
    let credentials = OAuthCredentials::from_store(store.clone()).expect("the credentials build");

    assert!(logout(&credentials).await.expect("the first logout"));
    assert!(store.load().await.expect("the store read").is_none());
    assert!(!logout(&credentials).await.expect("the second logout"));
}

#[tokio::test]
async fn logout_preserves_other_providers() {
    let directory = tempfile::tempdir().expect("a temp directory");
    let auth_path = directory.path().join("auth.json");
    let credentials = OAuthCredentials::at_path(&auth_path).expect("the credentials build");
    credentials
        .store()
        .save(&OAuthToken::new("access-1"))
        .await
        .expect("the saved token");
    let mut document = read_document(&auth_path);
    document["providers"]["Other"] = json!({ "tokens": { "access_token": "other" } });
    std::fs::write(
        &auth_path,
        serde_json::to_vec(&document).expect("the document"),
    )
    .expect("the rewritten store");

    assert!(logout(&credentials).await.expect("the logout"));

    let document = read_document(&auth_path);
    assert!(document["providers"].get("OpenAIAccount").is_none());
    assert_eq!(
        document["providers"]["Other"]["tokens"]["access_token"],
        "other"
    );
}

// --- credentials type -------------------------------------------------------

#[tokio::test]
async fn credentials_are_built_for_a_file_or_a_custom_store() {
    let directory = tempfile::tempdir().expect("a temp directory");
    let auth_path = directory.path().join("auth.json");

    let file = OAuthCredentials::at_path(&auth_path).expect("the file credentials");
    assert_eq!(file.auth_path(), Some(auth_path.as_path()));
    assert_ne!(file.client().base_url(), "");
    assert_eq!(file.skew(), super::store::REFRESH_SKEW);
    assert_eq!(
        FileTokenStore::new(&auth_path).path(),
        file.auth_path().expect("the file path")
    );

    let custom =
        OAuthCredentials::from_store(Arc::new(InMemoryTokenStore::new())).expect("the store");
    assert_eq!(custom.auth_path(), None);
    assert_eq!(custom.skew(), super::store::REFRESH_SKEW);

    let custom = custom.with_skew(std::time::Duration::from_secs(5));
    assert_eq!(custom.skew(), std::time::Duration::from_secs(5));
    assert!(
        custom
            .store()
            .load()
            .await
            .expect("the store read")
            .is_none(),
        "a fresh in-memory store is empty"
    );
}

#[test]
fn a_token_summary_never_carries_a_secret() {
    let token = OAuthToken::new("access-1")
        .with_refresh_token("refresh-1")
        .with_id_token("id-1")
        .with_expires_at(expiry_in(60))
        .with_scope("openid");
    let summary = super::credentials::TokenSummary::of(&token);

    assert_eq!(summary.expires_at, token.expires_at);
    assert!(summary.has_refresh_token);
    assert!(summary.has_id_token);
    assert_eq!(summary.scope.as_deref(), Some("openid"));
    let rendered = format!("{summary:?}");
    for secret in ["access-1", "refresh-1", "id-1"] {
        assert!(
            !rendered.contains(secret),
            "summary leaks {secret}: {rendered}"
        );
    }

    let empty = super::credentials::TokenSummary::of(&OAuthToken::new("access-2"));
    assert!(!empty.has_refresh_token);
    assert!(!empty.has_id_token);
    assert_eq!(empty.scope, None);
    assert_eq!(empty.token_type, "Bearer");
}
