//! Unit tests of the Grafana Cloud preset.
//!
//! `GRAFANA_*` has no Python counterpart in
//! `tests/unit_tests/extensions/tracer_otel/`, which only covers the generic
//! OTLP knobs; these tests hold the preset's own contract (endpoint derivation,
//! Basic credentials, environment precedence, and the collector deferral).
//!
//! Environment tests hold a process-wide lock and remove every variable on
//! entry and on drop, so they neither observe nor leak host configuration.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::{Mutex, MutexGuard, PoisonError};

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use openjiuwen_common::error::ErrorKind;

use super::{
    GrafanaAgentConfig, GrafanaCloudConfig, GRAFANA_API_TOKEN, GRAFANA_INSTANCE_ID,
    GRAFANA_OTLP_ENDPOINT, GRAFANA_OTLP_ZONE, GRAFANA_ZONE,
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
fn agent_preset_defers_to_an_external_collector() {
    let agent = GrafanaAgentConfig::new("http://localhost:4318").with_service_name("checkout");
    assert_eq!(agent.endpoint(), "http://localhost:4318");
    assert_eq!(agent.service_name(), Some("checkout"));

    let error = agent.otel_config().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.message().contains("collector"), "{error}");
    assert!(error.message().contains("Grafana Agent"), "{error}");
    assert!(error.message().contains("http://localhost:4318"), "{error}");
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
fn agent_preset_exporter_reports_the_collector_requirement() {
    let error = GrafanaAgentConfig::new("http://localhost:4318")
        .exporter()
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
}
