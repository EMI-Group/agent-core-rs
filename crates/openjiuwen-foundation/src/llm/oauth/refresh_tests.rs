//! Refresh-response and failure-classification tests for the `OpenAI` account
//! OAuth client.
//!
//! These exercise the precedence a refresh response is merged with and the three
//! failure outcomes the client distinguishes (rate limited, re-login required,
//! plain provider failure), always against a [`MockServer`] on loopback.

use chrono::{TimeDelta, Utc};
use openjiuwen_common::error::ErrorKind;
use openjiuwen_common::security::REDACTED;
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::store::OAuthToken;
use super::support::{expiry_in, jwt, jwt_expiry, oauth_client, refreshable, secret};

#[tokio::test]
async fn refresh_without_an_expiry_keeps_the_previous_expires_at() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "access_token": "new-access"
        })))
        .expect(1)
        .mount(&server)
        .await;

    let previous = OAuthToken::new("old-access")
        .with_refresh_token("old-refresh")
        .with_expires_at(expiry_in(3600));
    let refreshed = oauth_client(&server)
        .refresh(&previous)
        .await
        .expect("the refreshed token");

    assert_eq!(refreshed.expires_at, previous.expires_at);
    assert_eq!(
        secret(refreshed.refresh_token.as_ref()),
        Some("old-refresh"),
        "an omitted refresh token keeps the previous one"
    );
}

#[tokio::test]
async fn refresh_falls_back_to_the_jwt_claim_of_the_new_access_token() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "access_token": jwt(json!({ "exp": 1_900_000_000_u64 }))
        })))
        .expect(1)
        .mount(&server)
        .await;

    let previous = OAuthToken::new("old-access").with_refresh_token("old-refresh");
    assert!(
        previous.expires_at.is_none(),
        "the previous expiry is unknown"
    );
    let refreshed = oauth_client(&server)
        .refresh(&previous)
        .await
        .expect("the refreshed token");

    assert_eq!(refreshed.expires_at, Some(jwt_expiry()));
}

#[tokio::test]
async fn refresh_prefers_the_response_expiry_over_the_previous_one() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "access_token": "new-access",
            "expires_in": 60
        })))
        .expect(1)
        .mount(&server)
        .await;

    let previous = OAuthToken::new("old-access")
        .with_refresh_token("old-refresh")
        .with_expires_at(expiry_in(3600));
    let refreshed = oauth_client(&server)
        .refresh(&previous)
        .await
        .expect("the refreshed token");

    let remaining = refreshed
        .expires_at
        .expect("the response lifetime is honoured")
        .signed_duration_since(Utc::now());
    assert!(
        remaining < TimeDelta::try_seconds(120).expect("a duration"),
        "remaining: {remaining}"
    );
    assert_ne!(refreshed.expires_at, previous.expires_at);
}

#[tokio::test]
async fn refresh_carries_the_previous_token_type_over() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "access_token": "new-access",
            "expires_in": 3600
        })))
        .expect(1)
        .mount(&server)
        .await;

    let previous = OAuthToken::new("old-access")
        .with_refresh_token("old-refresh")
        .with_token_type("MAC");
    let refreshed = oauth_client(&server)
        .refresh(&previous)
        .await
        .expect("the refreshed token");

    assert_eq!(
        refreshed.token_type, "MAC",
        "an omitted token_type is carried over instead of resetting to Bearer"
    );
}

#[tokio::test]
async fn refresh_honours_a_token_type_the_response_reports() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "access_token": "new-access",
            "token_type": "bearer",
            "expires_in": 3600
        })))
        .mount(&server)
        .await;

    let refreshed = oauth_client(&server)
        .refresh(
            &OAuthToken::new("old-access")
                .with_refresh_token("old-refresh")
                .with_token_type("MAC"),
        )
        .await
        .expect("the refreshed token");

    assert_eq!(refreshed.token_type, "bearer");
}

// --- failure classification -------------------------------------------------

#[tokio::test]
async fn refresh_reports_rate_limiting_without_relogin() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(
            ResponseTemplate::new(429)
                .set_body_json(json!({ "error": { "code": "rate_limit_exceeded" } })),
        )
        .mount(&server)
        .await;

    let error = oauth_client(&server)
        .refresh(&refreshable())
        .await
        .expect_err("the endpoint is rate limited");

    assert_eq!(error.kind(), ErrorKind::Llm);
    let message = error.to_string();
    assert!(message.contains("rate limited"), "message: {message}");
    assert!(
        message.contains("still valid") && message.contains("no re-login is required"),
        "rate limiting must not read as a credential failure: {message}"
    );
}

#[tokio::test]
async fn refresh_requires_a_relogin_for_an_invalid_grant() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(ResponseTemplate::new(401).set_body_json(json!({
            "error": { "code": "invalid_grant", "message": "refresh token revoked" }
        })))
        .mount(&server)
        .await;

    let error = oauth_client(&server)
        .refresh(&refreshable())
        .await
        .expect_err("the grant is invalid");

    assert_eq!(error.kind(), ErrorKind::Llm);
    let message = error.to_string();
    assert!(message.contains("HTTP 401"), "message: {message}");
    assert!(message.contains("invalid_grant"), "message: {message}");
    assert!(
        message.contains("a re-login is required") && message.contains("device authorization flow"),
        "the caller must be told to sign in again: {message}"
    );
    assert!(
        !message.contains("still valid") && !message.contains("refresh token revoked"),
        "the vendor prose stays redacted: {message}"
    );
    assert!(message.contains(REDACTED), "message: {message}");
}

#[tokio::test]
async fn refresh_requires_a_relogin_for_an_unauthorized_status_without_a_known_code() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(ResponseTemplate::new(403).set_body_string(r#"{"error":"forbidden"}"#))
        .mount(&server)
        .await;

    let error = oauth_client(&server)
        .refresh(&refreshable())
        .await
        .expect_err("the endpoint forbids the request");

    let message = error.to_string();
    assert!(message.contains("HTTP 403"), "message: {message}");
    assert!(
        message.contains("a re-login is required"),
        "a 403 invalidates the credentials: {message}"
    );
    assert!(
        !message.contains("forbidden"),
        "an unknown vendor code is never echoed: {message}"
    );
}

#[tokio::test]
async fn refresh_requires_a_relogin_for_a_reused_refresh_token() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "error": { "type": "refresh_token_reused", "message": "already used" }
        })))
        .mount(&server)
        .await;

    let error = oauth_client(&server)
        .refresh(&refreshable())
        .await
        .expect_err("the refresh token was already rotated");

    let message = error.to_string();
    assert!(message.contains("HTTP 400"), "message: {message}");
    assert!(
        message.contains("refresh_token_reused"),
        "message: {message}"
    );
    assert!(
        message.contains("a re-login is required"),
        "message: {message}"
    );
}

#[tokio::test]
async fn refresh_reports_a_provider_failure_without_relogin() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(ResponseTemplate::new(500).set_body_json(json!({
            "error": { "code": "server_error", "message": "boom" }
        })))
        .mount(&server)
        .await;

    let error = oauth_client(&server)
        .refresh(&refreshable())
        .await
        .expect_err("the provider fails");

    assert_eq!(error.kind(), ErrorKind::Llm);
    let message = error.to_string();
    assert!(message.contains("HTTP 500"), "message: {message}");
    assert!(
        !message.contains("a re-login is required"),
        "a server error must not invalidate the credentials: {message}"
    );
    assert!(
        !message.contains("server_error") && !message.contains("boom"),
        "the vendor body stays redacted: {message}"
    );
    assert!(message.contains(REDACTED), "message: {message}");
}

#[tokio::test]
async fn refresh_reports_an_unusable_success_body_as_a_relogin() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/oauth/token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "token_type": "Bearer"
        })))
        .mount(&server)
        .await;

    let error = oauth_client(&server)
        .refresh(&refreshable())
        .await
        .expect_err("the body carries no access token");

    assert_eq!(error.kind(), ErrorKind::Llm);
    let message = error.to_string();
    assert!(
        message.contains("not a token") && message.contains("a re-login is required"),
        "message: {message}"
    );
}
