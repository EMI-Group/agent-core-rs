//! Token-shape tests: expiry precedence, serialization, and `Debug` redaction.
//!
//! The expiry rules are pure data transformations, so these are plain unit tests
//! with no server; the refresh path is covered by `refresh_tests`.

use std::time::Duration;

use chrono::{DateTime, TimeDelta, Utc};
use openjiuwen_common::security::REDACTED;
use serde_json::json;

use super::store::{parse_token_payload, DeviceCode, OAuthToken, REFRESH_SKEW};
use super::support::{expiry_in, jwt, jwt_expiry, secret};

// --- token serialization ----------------------------------------------------

#[test]
fn token_deserializes_expires_in_into_an_expiry() {
    let token: OAuthToken = serde_json::from_value(json!({
        "access_token": "access-1",
        "refresh_token": "refresh-1",
        "id_token": "id-1",
        "token_type": "Bearer",
        "scope": "openid",
        "expires_in": 3600,
        "unknown_field": { "ignored": true }
    }))
    .expect("the token payload");

    assert_eq!(token.access_token.expose_secret(), "access-1");
    assert_eq!(secret(token.refresh_token.as_ref()), Some("refresh-1"));
    assert_eq!(secret(token.id_token.as_ref()), Some("id-1"));
    assert_eq!(token.scope.as_deref(), Some("openid"));
    let remaining = token
        .expires_at
        .expect("the payload carries a lifetime")
        .signed_duration_since(Utc::now());
    assert!(remaining > TimeDelta::try_seconds(3_500).expect("a duration"));
    assert!(remaining <= TimeDelta::try_seconds(3_600).expect("a duration"));
    assert!(!token.is_expired(REFRESH_SKEW));
}

#[test]
fn token_accepts_epoch_and_rfc3339_expiries() {
    let epoch: OAuthToken =
        serde_json::from_value(json!({ "access_token": "a", "expires_at": 1_900_000_000 }))
            .expect("an epoch expiry");
    assert_eq!(epoch.expires_at, DateTime::from_timestamp(1_900_000_000, 0));

    let rfc3339: OAuthToken = serde_json::from_value(
        json!({ "access_token": "a", "expires_at": "2030-01-01T00:00:00Z" }),
    )
    .expect("an RFC 3339 expiry");
    assert_eq!(
        rfc3339.expires_at,
        DateTime::from_timestamp(1_893_456_000, 0)
    );

    let stringy: OAuthToken =
        serde_json::from_value(json!({ "access_token": "a", "expires_in": "1800" }))
            .expect("a numeric string lifetime");
    let remaining = stringy
        .expires_at
        .expect("the lifetime is honoured")
        .signed_duration_since(Utc::now());
    assert!(remaining > TimeDelta::try_seconds(1_700).expect("a duration"));
}

#[test]
fn token_serializes_the_canonical_fields() {
    let token = OAuthToken::new("access-1")
        .with_refresh_token("refresh-1")
        .with_scope("openid")
        .with_expires_at(DateTime::from_timestamp(1_900_000_000, 0).expect("a timestamp"));

    let value = serde_json::to_value(&token).expect("the serialized token");
    assert_eq!(
        value,
        json!({
            "access_token": "access-1",
            "refresh_token": "refresh-1",
            "expires_at": 1_900_000_000,
            "token_type": "Bearer",
            "scope": "openid"
        })
    );

    let minimal = serde_json::to_value(OAuthToken::new("access-2")).expect("the serialized token");
    assert_eq!(
        minimal,
        json!({ "access_token": "access-2", "token_type": "Bearer" })
    );
}

#[test]
fn token_round_trips_through_json() {
    let original = OAuthToken::new("access-1")
        .with_refresh_token("refresh-1")
        .with_id_token("id-1")
        .with_expires_at(expiry_in(3600));

    let restored: OAuthToken =
        serde_json::from_value(serde_json::to_value(&original).expect("the serialized token"))
            .expect("the deserialized token");

    assert_eq!(restored.access_token, original.access_token);
    assert_eq!(restored.refresh_token, original.refresh_token);
    assert_eq!(restored.id_token, original.id_token);
    assert_eq!(restored.expires_at, original.expires_at);
    assert_eq!(restored.token_type, original.token_type);
    assert_eq!(restored.scope, original.scope);
}

#[test]
fn token_without_a_resolvable_expiry_is_treated_as_expired() {
    let token = OAuthToken::new("access-1");
    assert!(token.expires_at.is_none());
    assert!(
        token.is_expired(Duration::ZERO),
        "an unresolvable expiry is stale, so the refresh flow runs on use"
    );
    assert!(token.is_expired(Duration::from_secs(86_400)));

    let expired = OAuthToken::new("access-1").with_expires_at(expiry_in(-10));
    assert!(expired.is_expired(Duration::ZERO));

    let within_skew = OAuthToken::new("access-1").with_expires_at(expiry_in(60));
    assert!(within_skew.is_expired(REFRESH_SKEW));
    assert!(!within_skew.is_expired(Duration::from_secs(10)));

    let valid = OAuthToken::new("access-1").with_expires_at(expiry_in(3600));
    assert!(!valid.is_expired(REFRESH_SKEW), "a future expiry is valid");
}

#[test]
fn token_expiry_prefers_expires_at_over_the_lifetime_and_the_jwt_claim() {
    let token: OAuthToken = serde_json::from_value(json!({
        "access_token": jwt(json!({ "exp": 1_900_000_000_u64 })),
        "expires_at": 1_800_000_000,
        "expires_in": 60
    }))
    .expect("the token payload");

    assert_eq!(
        token.expires_at,
        DateTime::from_timestamp(1_800_000_000, 0),
        "an explicit expires_at wins"
    );
}

#[test]
fn token_expiry_uses_the_lifetime_before_the_jwt_claim() {
    let token: OAuthToken = serde_json::from_value(json!({
        "access_token": jwt(json!({ "exp": 1_900_000_000_u64 })),
        "expires_in": 60
    }))
    .expect("the token payload");

    let remaining = token
        .expires_at
        .expect("the lifetime is honoured")
        .signed_duration_since(Utc::now());
    assert!(
        remaining > TimeDelta::zero()
            && remaining <= TimeDelta::try_seconds(60).expect("a duration"),
        "remaining: {remaining}"
    );
    assert_ne!(token.expires_at, Some(jwt_expiry()));
}

#[test]
fn token_expiry_falls_back_to_the_jwt_claim() {
    let encoded = jwt(json!({ "exp": 1_900_000_000_u64 }));
    let token: OAuthToken =
        serde_json::from_value(json!({ "access_token": encoded })).expect("the JWT token");
    assert_eq!(token.expires_at, Some(jwt_expiry()));
    assert_eq!(token.effective_expires_at(), Some(jwt_expiry()));

    // A padded payload segment decodes too: the padding is stripped first.
    let mut segments = encoded.split('.');
    let header = segments.next().expect("the header segment");
    let payload = segments.next().expect("the payload segment");
    let padded = format!("{header}.{payload}==.signature");
    let token: OAuthToken =
        serde_json::from_value(json!({ "access_token": padded })).expect("the padded JWT token");
    assert_eq!(token.expires_at, Some(jwt_expiry()));
}

#[test]
fn token_expiry_is_unknown_when_nothing_resolves() {
    let opaque = OAuthToken::new("opaque-access-token");
    assert_eq!(opaque.effective_expires_at(), None);
    assert!(
        opaque.is_expired(REFRESH_SKEW),
        "an unknown expiry counts as expired"
    );
    assert!(opaque.is_expired(Duration::from_secs(86_400)));

    for access_token in [
        jwt(json!({ "sub": "user" })),
        "header.not*base64!.signature".to_owned(),
        "two.parts".to_owned(),
        "single-segment".to_owned(),
    ] {
        let token: OAuthToken = serde_json::from_value(json!({ "access_token": access_token }))
            .expect("the token payload");
        assert_eq!(
            token.expires_at,
            None,
            "no resolvable expiry for {}",
            token.access_token.expose_secret()
        );
        assert!(
            token.is_expired(REFRESH_SKEW),
            "an unknown expiry must trigger the refresh flow"
        );
    }
}

#[test]
fn token_treats_a_zero_or_negative_lifetime_as_already_expired() {
    for lifetime in [json!(0), json!(-600), json!("0")] {
        let token: OAuthToken =
            serde_json::from_value(json!({ "access_token": "opaque", "expires_in": lifetime }))
                .expect("the token payload");
        assert!(
            token.expires_at.is_some(),
            "a reported lifetime is never dropped: {lifetime}"
        );
        assert!(
            token.is_expired(REFRESH_SKEW),
            "a non-positive lifetime means the token is already stale: {lifetime}"
        );
    }
}

#[test]
fn token_payload_takes_the_previous_expiry_before_the_jwt_claim() {
    let previous = OAuthToken::new("old-access")
        .with_expires_at(DateTime::from_timestamp(1_800_000_000, 0).expect("a timestamp"));
    let payload = serde_json::to_vec(&json!({
        "access_token": jwt(json!({ "exp": 1_900_000_000_u64 }))
    }))
    .expect("the payload");

    let token = parse_token_payload(&payload, Some(&previous)).expect("the token");

    assert_eq!(token.expires_at, previous.expires_at);
}

#[test]
fn token_payload_uses_the_jwt_claim_when_no_previous_expiry_exists() {
    let previous = OAuthToken::new("old-access");
    let payload = serde_json::to_vec(&json!({
        "access_token": jwt(json!({ "exp": 1_900_000_000_u64 }))
    }))
    .expect("the payload");

    let token = parse_token_payload(&payload, Some(&previous)).expect("the token");

    assert_eq!(token.expires_at, Some(jwt_expiry()));
}

#[test]
fn token_debug_redacts_every_secret() {
    let token = OAuthToken::new("super-secret-access")
        .with_refresh_token("super-secret-refresh")
        .with_id_token("super-secret-id");

    let rendered = format!("{token:?}");
    assert!(rendered.contains("OAuthToken"), "rendered: {rendered}");
    assert!(rendered.contains(REDACTED), "rendered: {rendered}");
    assert!(!rendered.contains("super-secret"), "rendered: {rendered}");
    assert!(
        !format!("{:?}", token.refresh_token).contains("super-secret-refresh"),
        "a nested option is redacted too"
    );
}

#[test]
fn device_code_is_tolerant_of_missing_and_string_fields() {
    let minimal: DeviceCode =
        serde_json::from_value(json!({ "user_code": "ABCD", "device_auth_id": "device-1" }))
            .expect("the minimal device code");
    assert_eq!(minimal.interval, 5);
    assert_eq!(minimal.expires_in, 0);

    let stringy: DeviceCode = serde_json::from_value(json!({
        "user_code": "ABCD",
        "device_auth_id": "device-1",
        "interval": "10",
        "expires_in": "600"
    }))
    .expect("the tolerant device code");
    assert_eq!(
        stringy,
        DeviceCode::new("ABCD", "device-1")
            .with_interval(10)
            .with_expires_in(600)
    );
}
