//! Unit tests of the Grafana Cloud and Grafana Agent/Alloy presets.
//!
//! `GRAFANA_*` has no Python counterpart in
//! `tests/unit_tests/extensions/tracer_otel/`, which only covers the generic
//! OTLP knobs; these tests hold the presets' own contract (endpoint derivation,
//! Basic credentials, environment precedence, and the local collector push).
//!
//! Environment tests hold a process-wide lock and remove every variable on
//! entry and on drop, so they neither observe nor leak host configuration.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard, PoisonError};

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use openjiuwen_common::error::ErrorKind;

use super::{
    GrafanaAgentConfig, GrafanaCloudConfig, DEFAULT_LOCAL_OTLP_ENDPOINT, GRAFANA_API_TOKEN,
    GRAFANA_INSTANCE_ID, GRAFANA_OTLP_ENDPOINT, GRAFANA_OTLP_ZONE, GRAFANA_ZONE,
};
use crate::otel::config::{ExporterType, OtlpProtocol};

/// Serialises the tests that mutate `GRAFANA_*`.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Every variable [`GrafanaCloudConfig::from_env`] reads.
const ENV_VARS: [&str; 5] = [
    GRAFANA_INSTANCE_ID,
    GRAFANA_API_TOKEN,
    GRAFANA_ZONE,
    GRAFANA_OTLP_ZONE,
    GRAFANA_OTLP_ENDPOINT,
];

/// Holds [`ENV_LOCK`] and keeps the `GRAFANA_*` variables clean.
struct EnvGuard {
    /// The lock guard; dropped last.
    _lock: MutexGuard<'static, ()>,
}

impl EnvGuard {
    /// Acquires the lock, recovering it when another test panicked.
    fn new() -> Self {
        let lock = ENV_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        clear_env();
        Self { _lock: lock }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        clear_env();
    }
}

/// Removes every `GRAFANA_*` variable.
fn clear_env() {
    for name in ENV_VARS {
        std::env::remove_var(name);
    }
}

/// Sets one `GRAFANA_*` variable for the rest of the test.
fn set_env(name: &str, value: &str) {
    std::env::set_var(name, value);
}

#[test]
fn endpoint_derives_the_gateway_url_from_a_zone_slug() {
    let config = GrafanaCloudConfig::new("123", "glc_token", "prod-us-east-0");
    assert_eq!(
        config.endpoint(),
        "https://otlp-gateway-prod-us-east-0.grafana.net/otlp",
    );
}

#[test]
fn endpoint_accepts_a_zone_that_already_carries_the_gateway_prefix() {
    let slug = GrafanaCloudConfig::new("123", "glc_token", "prod-us-east-0");
    let label = GrafanaCloudConfig::new("123", "glc_token", "otlp-gateway-prod-us-east-0");
    let host = GrafanaCloudConfig::new(
        "123",
        "glc_token",
        "otlp-gateway-prod-us-east-0.grafana.net",
    );
    let url = GrafanaCloudConfig::new(
        "123",
        "glc_token",
        "https://otlp-gateway-prod-us-east-0.grafana.net/otlp/",
    );
    let expected = "https://otlp-gateway-prod-us-east-0.grafana.net/otlp";
    for config in [slug, label, host, url] {
        assert_eq!(config.endpoint(), expected, "zone: {:?}", config.zone());
    }
}

#[test]
fn endpoint_override_wins_over_the_zone() {
    let config = GrafanaCloudConfig::new("123", "glc_token", "prod-us-east-0")
        .with_otlp_endpoint("https://collector.example.com/otlp/");
    assert_eq!(config.endpoint(), "https://collector.example.com/otlp");
    config.validate().unwrap();
}

#[test]
fn authorization_is_the_standard_base64_of_instance_and_token() {
    let config = GrafanaCloudConfig::new("123", "glc_token", "prod-us-east-0");
    assert_eq!(
        config.authorization(),
        format!("Basic {}", STANDARD.encode("123:glc_token")),
    );
    assert_eq!(config.authorization(), "Basic MTIzOmdsY190b2tlbg==");
}

#[test]
fn otel_config_selects_the_http_otlp_exporter() {
    let config =
        GrafanaCloudConfig::new("123", "glc_token", "prod-eu-west-2").with_service_name("checkout");
    let otel = config.otel_config().unwrap();
    assert_eq!(otel.exporter_type, ExporterType::Otlp);
    assert_eq!(otel.protocol, OtlpProtocol::Http);
    assert_eq!(
        otel.exporter_endpoint.as_deref(),
        Some("https://otlp-gateway-prod-eu-west-2.grafana.net/otlp"),
    );
    assert_eq!(otel.headers.len(), 1);
    assert_eq!(otel.headers["authorization"], config.authorization());
    assert_eq!(otel.service_name, "checkout");
    assert_eq!(otel.service_version, None);
}

#[test]
fn debug_redacts_the_token() {
    let config = GrafanaCloudConfig::new("123", "glc_token", "prod-us-east-0");
    let rendered = format!("{config:?}");
    assert!(rendered.contains("<redacted>"), "{rendered}");
    assert!(!rendered.contains("glc_token"), "{rendered}");
}

#[test]
fn validate_rejects_an_empty_instance_id() {
    let config = GrafanaCloudConfig::new("", "glc_token", "prod-us-east-0");
    let error = config.validate().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("instance_id"), "{error}");
    assert!(error.message().contains(GRAFANA_INSTANCE_ID), "{error}");
}

#[test]
fn validate_rejects_an_empty_token() {
    let config = GrafanaCloudConfig::new("123", "   ", "prod-us-east-0");
    let error = config.validate().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("token"), "{error}");
    assert!(error.message().contains(GRAFANA_API_TOKEN), "{error}");
}

#[test]
fn validate_rejects_a_zone_that_could_smuggle_a_path() {
    for zone in [
        "prod-us-east-0/../../admin",
        "prod us-east-0",
        "prod-us-east-0?token=1",
        "prod-us-east-0#fragment",
        "prod--us-east-0",
        "https://prod-us-east-0",
    ] {
        let config = GrafanaCloudConfig::new("123", "glc_token", zone);
        let error = config.validate().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "zone: {zone}");
        assert!(error.message().contains("zone"), "zone: {zone} ({error})");
        assert_eq!(config.endpoint(), "", "zone: {zone}");
    }
}

#[test]
fn validate_rejects_an_empty_zone_without_an_endpoint_override() {
    let config = GrafanaCloudConfig::new("123", "glc_token", "");
    let error = config.validate().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains(GRAFANA_ZONE), "{error}");
    assert!(error.message().contains(GRAFANA_OTLP_ENDPOINT), "{error}");
}

#[test]
fn validate_rejects_a_relative_endpoint_override() {
    let config =
        GrafanaCloudConfig::new("123", "glc_token", "prod-us-east-0").with_otlp_endpoint("/otlp");
    let error = config.validate().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("endpoint"), "{error}");
}

#[test]
fn from_env_reads_the_cloud_credentials() {
    let _guard = EnvGuard::new();
    set_env(GRAFANA_INSTANCE_ID, "123");
    set_env(GRAFANA_API_TOKEN, "glc_token");
    set_env(GRAFANA_ZONE, "prod-us-east-0");

    let config = GrafanaCloudConfig::from_env().unwrap();
    assert_eq!(config.zone(), "prod-us-east-0");
    assert_eq!(
        config.endpoint(),
        "https://otlp-gateway-prod-us-east-0.grafana.net/otlp",
    );
    assert_eq!(config.service_name(), "openjiuwen");
}

#[test]
fn from_env_prefers_grafana_zone_over_the_alias() {
    let _guard = EnvGuard::new();
    set_env(GRAFANA_INSTANCE_ID, "123");
    set_env(GRAFANA_API_TOKEN, "glc_token");
    set_env(GRAFANA_ZONE, "prod-us-east-0");
    set_env(GRAFANA_OTLP_ZONE, "prod-eu-west-2");

    let config = GrafanaCloudConfig::from_env().unwrap();
    assert_eq!(config.zone(), "prod-us-east-0");
    assert!(config.endpoint().contains("prod-us-east-0"), "{config:?}");
}

#[test]
fn from_env_falls_back_to_the_zone_alias() {
    let _guard = EnvGuard::new();
    set_env(GRAFANA_INSTANCE_ID, "123");
    set_env(GRAFANA_API_TOKEN, "glc_token");
    set_env(GRAFANA_OTLP_ZONE, "prod-eu-west-2");

    let config = GrafanaCloudConfig::from_env().unwrap();
    assert_eq!(config.zone(), "prod-eu-west-2");
}

#[test]
fn from_env_endpoint_override_wins_over_the_zone() {
    let _guard = EnvGuard::new();
    set_env(GRAFANA_INSTANCE_ID, "123");
    set_env(GRAFANA_API_TOKEN, "glc_token");
    set_env(GRAFANA_ZONE, "prod-us-east-0");
    set_env(GRAFANA_OTLP_ENDPOINT, "https://collector.internal:4318");

    let config = GrafanaCloudConfig::from_env().unwrap();
    assert_eq!(config.endpoint(), "https://collector.internal:4318");
}

#[test]
fn from_env_accepts_an_endpoint_without_a_zone() {
    let _guard = EnvGuard::new();
    set_env(GRAFANA_INSTANCE_ID, "123");
    set_env(GRAFANA_API_TOKEN, "glc_token");
    set_env(
        GRAFANA_OTLP_ENDPOINT,
        "https://collector.internal:4318/otlp",
    );

    let config = GrafanaCloudConfig::from_env().unwrap();
    assert_eq!(config.endpoint(), "https://collector.internal:4318/otlp");
}

#[test]
fn from_env_names_a_missing_variable() {
    let _guard = EnvGuard::new();
    let error = GrafanaCloudConfig::from_env().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains(GRAFANA_INSTANCE_ID), "{error}");

    set_env(GRAFANA_INSTANCE_ID, "123");
    let error = GrafanaCloudConfig::from_env().unwrap_err();
    assert!(error.message().contains(GRAFANA_API_TOKEN), "{error}");

    // A blank value counts as unset (Python `os.getenv(...) or None`).
    set_env(GRAFANA_API_TOKEN, "  ");
    let error = GrafanaCloudConfig::from_env().unwrap_err();
    assert!(error.message().contains(GRAFANA_API_TOKEN), "{error}");
}

#[test]
fn from_env_requires_a_zone_without_an_endpoint() {
    let _guard = EnvGuard::new();
    set_env(GRAFANA_INSTANCE_ID, "123");
    set_env(GRAFANA_API_TOKEN, "glc_token");

    let error = GrafanaCloudConfig::from_env().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains(GRAFANA_ZONE), "{error}");
}

#[test]
fn agent_preset_selects_the_local_http_exporter() {
    let agent = GrafanaAgentConfig::new("http://127.0.0.1:4318").with_service_name("checkout");
    assert_eq!(agent.endpoint(), "http://127.0.0.1:4318");
    assert_eq!(agent.service_name(), Some("checkout"));
    assert!(agent.headers().is_empty());

    let otel = agent.otel_config().unwrap();
    assert_eq!(otel.exporter_type, ExporterType::Otlp);
    assert_eq!(otel.protocol, OtlpProtocol::Http);
    assert_eq!(
        otel.exporter_endpoint.as_deref(),
        Some("http://127.0.0.1:4318"),
    );
    assert_eq!(otel.service_name, "checkout");
    assert_eq!(otel.service_version, None);
    assert!(otel.headers.is_empty());
}

#[test]
fn agent_preset_defaults_to_the_local_collector_endpoint() {
    for agent in [
        GrafanaAgentConfig::default(),
        GrafanaAgentConfig::new("   "),
    ] {
        let otel = agent.otel_config().unwrap();
        assert_eq!(
            otel.exporter_endpoint.as_deref(),
            Some(DEFAULT_LOCAL_OTLP_ENDPOINT),
            "endpoint: {:?}",
            agent.endpoint(),
        );
        // The Grafana Agent/Alloy OTLP/HTTP default port, shared with the
        // generic exporter.
        assert_eq!(DEFAULT_LOCAL_OTLP_ENDPOINT, "http://localhost:4318");
        assert_eq!(otel.service_name, "openjiuwen");
        assert_eq!(otel.protocol, OtlpProtocol::Http);
    }
}

#[test]
fn agent_preset_keeps_a_custom_endpoint_without_a_trailing_slash() {
    let otel = GrafanaAgentConfig::new(" https://alloy.internal:4318/ ")
        .otel_config()
        .unwrap();
    assert_eq!(
        otel.exporter_endpoint.as_deref(),
        Some("https://alloy.internal:4318"),
    );
}

#[test]
fn agent_preset_carries_headers() {
    let agent = GrafanaAgentConfig::new("http://127.0.0.1:4318")
        .with_header("authorization", "Bearer local")
        .with_headers(BTreeMap::from([
            ("x-tenant".to_owned(), "acme".to_owned()),
            ("x-scope".to_owned(), "traces".to_owned()),
        ]))
        .with_header("x-tenant", "rotated");
    assert_eq!(agent.headers().len(), 3);
    assert_eq!(agent.headers()["x-tenant"], "rotated");

    let otel = agent.otel_config().unwrap();
    assert_eq!(otel.headers.len(), 3);
    assert_eq!(otel.headers["authorization"], "Bearer local");
    assert_eq!(otel.headers["x-tenant"], "rotated");
    assert_eq!(otel.headers["x-scope"], "traces");
}

#[test]
fn agent_preset_rejects_an_unusable_endpoint() {
    for endpoint in [
        "/v1/traces",
        "127.0.0.1:4318",
        "ftp://127.0.0.1:4318",
        "http://",
    ] {
        let error = GrafanaAgentConfig::new(endpoint).otel_config().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "endpoint: {endpoint}");
        assert!(
            error.message().contains("endpoint"),
            "endpoint: {endpoint} ({error})"
        );
    }
}

#[test]
fn agent_preset_rejects_an_explicitly_blank_service_name() {
    let error = GrafanaAgentConfig::default()
        .with_service_name("   ")
        .otel_config()
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("service_name"), "{error}");
}

#[test]
fn agent_preset_rejects_invalid_headers() {
    let invalid_name = GrafanaAgentConfig::default()
        .with_header("bad name", "value")
        .otel_config()
        .unwrap_err();
    assert_eq!(invalid_name.kind(), ErrorKind::Validation);
    assert!(
        invalid_name.message().contains("header name"),
        "{invalid_name}"
    );

    let empty_name = GrafanaAgentConfig::default()
        .with_header("", "value")
        .otel_config()
        .unwrap_err();
    assert_eq!(empty_name.kind(), ErrorKind::Validation);
    assert!(empty_name.message().contains("header name"), "{empty_name}");

    let invalid_value = GrafanaAgentConfig::default()
        .with_header("x-tenant", "line\nbreak")
        .otel_config()
        .unwrap_err();
    assert_eq!(invalid_value.kind(), ErrorKind::Validation);
    assert!(
        invalid_value.message().contains("x-tenant"),
        "{invalid_value}"
    );
}

#[test]
fn agent_preset_round_trips_through_json() {
    let agent = GrafanaAgentConfig::new("http://alloy.internal:4318")
        .with_service_name("checkout")
        .with_header("x-tenant", "acme");
    let json = serde_json::to_value(&agent).unwrap();
    assert_eq!(
        json["endpoint"],
        serde_json::json!("http://alloy.internal:4318")
    );
    assert_eq!(json["service_name"], serde_json::json!("checkout"));
    assert_eq!(json["headers"], serde_json::json!({"x-tenant": "acme"}));

    let restored: GrafanaAgentConfig = serde_json::from_value(json).unwrap();
    assert_eq!(restored, agent);

    // A document written before the `headers` field existed still loads.
    let legacy: GrafanaAgentConfig = serde_json::from_value(serde_json::json!({
        "endpoint": "http://127.0.0.1:4318",
        "service_name": "legacy",
    }))
    .unwrap();
    assert_eq!(legacy.endpoint(), "http://127.0.0.1:4318");
    assert_eq!(legacy.service_name(), Some("legacy"));
    assert!(legacy.headers().is_empty());

    // An empty object is the default preset (local collector, default name).
    let empty: GrafanaAgentConfig = serde_json::from_value(serde_json::json!({})).unwrap();
    assert_eq!(empty, GrafanaAgentConfig::default());
}

#[cfg(feature = "otel")]
#[test]
fn exporter_posts_to_the_traces_signal_path() {
    let exporter = GrafanaCloudConfig::new("123", "glc_token", "prod-us-east-0")
        .exporter()
        .unwrap();
    assert_eq!(
        exporter.traces_url(),
        "https://otlp-gateway-prod-us-east-0.grafana.net/otlp/v1/traces",
    );
    assert!(exporter.traces_url().ends_with("/v1/traces"));
}

#[cfg(feature = "otel")]
#[test]
fn agent_preset_exporter_posts_to_the_local_traces_path() {
    let exporter = GrafanaAgentConfig::new("http://127.0.0.1:4318")
        .with_header("x-tenant", "acme")
        .exporter()
        .unwrap();
    assert_eq!(exporter.traces_url(), "http://127.0.0.1:4318/v1/traces");

    // The default preset targets the local collector as well.
    let default = GrafanaAgentConfig::default().exporter().unwrap();
    assert_eq!(
        default.traces_url(),
        format!("{DEFAULT_LOCAL_OTLP_ENDPOINT}/v1/traces"),
    );
}
