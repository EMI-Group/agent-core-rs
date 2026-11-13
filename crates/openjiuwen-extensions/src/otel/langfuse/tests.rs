//! Unit tests of the Langfuse preset.
//!
//! The preset covers the connection side of
//! `openjiuwen.extensions.observability.exporters.langfuse` (host, credentials,
//! and the OTLP endpoint); the ingestion client has its own suite in
//! `client/tests.rs`.
//!
//! Environment tests hold a process-wide lock and remove every variable on
//! entry and on drop, so they neither observe nor leak host configuration.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::{Mutex, MutexGuard, PoisonError};

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use openjiuwen_common::error::ErrorKind;

use super::{
    LangfuseClient, LangfuseConfig, DEFAULT_HOST, LANGFUSE_BASEURL, LANGFUSE_HOST,
    LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY,
};
use crate::otel::config::{ExporterType, OtlpProtocol};

/// Serialises the tests that mutate `LANGFUSE_*`.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Every variable [`LangfuseConfig::from_env`] reads.
const ENV_VARS: [&str; 4] = [
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_HOST,
    LANGFUSE_BASEURL,
];

/// Holds [`ENV_LOCK`] and keeps the `LANGFUSE_*` variables clean.
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

/// Removes every `LANGFUSE_*` variable.
fn clear_env() {
    for name in ENV_VARS {
        std::env::remove_var(name);
    }
}

/// Sets one `LANGFUSE_*` variable for the rest of the test.
fn set_env(name: &str, value: &str) {
    std::env::set_var(name, value);
}

#[test]
fn endpoint_defaults_to_langfuse_cloud() {
    let config = LangfuseConfig::new("pk-lf-public", "sk-lf-secret");
    assert_eq!(config.host(), DEFAULT_HOST);
    assert_eq!(
        config.endpoint(),
        "https://cloud.langfuse.com/api/public/otel"
    );
    assert_eq!(
        config.ingestion_url(),
        "https://cloud.langfuse.com/api/public/ingestion",
    );
}

#[test]
fn endpoint_normalises_trailing_slashes_and_pasted_rest_paths() {
    let expected = "https://langfuse.internal/api/public/otel";
    let expected_ingestion = "https://langfuse.internal/api/public/ingestion";
    for host in [
        "https://langfuse.internal",
        "https://langfuse.internal/",
        "https://langfuse.internal//",
        " https://langfuse.internal ",
        "https://langfuse.internal/api/public",
        "https://langfuse.internal/api/public/otel",
        "https://langfuse.internal/api/public/ingestion",
    ] {
        let config = LangfuseConfig::new("pk-lf-public", "sk-lf-secret").with_host(host);
        assert_eq!(config.endpoint(), expected, "host: {host}");
        assert_eq!(config.ingestion_url(), expected_ingestion, "host: {host}");
        config.validate().unwrap();
    }
}

#[test]
fn authorization_is_the_standard_base64_of_public_and_secret_key() {
    let config = LangfuseConfig::new("pk-lf-public", "sk-lf-secret");
    assert_eq!(
        config.authorization(),
        format!("Basic {}", STANDARD.encode("pk-lf-public:sk-lf-secret")),
    );
}

#[test]
fn otel_config_selects_the_http_otlp_exporter() {
    let config = LangfuseConfig::new("pk-lf-public", "sk-lf-secret")
        .with_host("https://langfuse.internal")
        .with_service_name("checkout");
    let otel = config.otel_config().unwrap();
    assert_eq!(otel.exporter_type, ExporterType::Otlp);
    assert_eq!(otel.protocol, OtlpProtocol::Http);
    assert_eq!(
        otel.exporter_endpoint.as_deref(),
        Some("https://langfuse.internal/api/public/otel"),
    );
    assert_eq!(otel.headers.len(), 1);
    assert_eq!(otel.headers["authorization"], config.authorization());
    assert_eq!(otel.service_name, "checkout");
}

#[test]
fn debug_redacts_the_secret_key() {
    let config = LangfuseConfig::new("pk-lf-public", "sk-lf-secret");
    let rendered = format!("{config:?}");
    assert!(rendered.contains("<redacted>"), "{rendered}");
    assert!(!rendered.contains("sk-lf-secret"), "{rendered}");
}

#[test]
fn validate_rejects_blank_keys() {
    let error = LangfuseConfig::new("", "sk-lf-secret")
        .validate()
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("public_key"), "{error}");
    assert!(error.message().contains(LANGFUSE_PUBLIC_KEY), "{error}");

    let error = LangfuseConfig::new("pk-lf-public", " ")
        .validate()
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("secret_key"), "{error}");
    assert!(error.message().contains(LANGFUSE_SECRET_KEY), "{error}");
}

#[test]
fn validate_rejects_a_host_that_is_not_an_absolute_url() {
    let config = LangfuseConfig::new("pk-lf-public", "sk-lf-secret").with_host("langfuse.internal");
    let error = config.validate().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("host"), "{error}");

    let config = LangfuseConfig::new("pk-lf-public", "sk-lf-secret").with_host("ftp://langfuse");
    let error = config.validate().unwrap_err();
    assert!(error.message().contains("http"), "{error}");
}

#[test]
fn client_construction_validates_the_preset() {
    let error = LangfuseClient::new(LangfuseConfig::new("", "sk-lf-secret")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);

    let client = LangfuseClient::new(LangfuseConfig::new("pk-lf-public", "sk-lf-secret")).unwrap();
    assert_eq!(client.max_batch_size(), 1000);
    assert_eq!(client.config().host(), DEFAULT_HOST);
    assert!(format!("{client:?}").contains("LangfuseClient"));
    let client = client.with_max_batch_size(0);
    assert_eq!(client.max_batch_size(), 1);
    assert_eq!(client.with_max_batch_size(7).max_batch_size(), 7);
}

#[test]
fn from_env_reads_the_project_keys() {
    let _guard = EnvGuard::new();
    set_env(LANGFUSE_PUBLIC_KEY, "pk-lf-public");
    set_env(LANGFUSE_SECRET_KEY, "sk-lf-secret");

    let config = LangfuseConfig::from_env().unwrap();
    assert_eq!(config.host(), DEFAULT_HOST);
    assert_eq!(config.service_name(), "openjiuwen");

    let client = LangfuseClient::from_env().unwrap();
    assert_eq!(client.config().host(), DEFAULT_HOST);
}

#[test]
fn from_env_prefers_langfuse_host_over_the_baseurl_alias() {
    let _guard = EnvGuard::new();
    set_env(LANGFUSE_PUBLIC_KEY, "pk-lf-public");
    set_env(LANGFUSE_SECRET_KEY, "sk-lf-secret");
    set_env(LANGFUSE_HOST, "https://langfuse.internal");
    set_env(LANGFUSE_BASEURL, "https://langfuse.ignored");

    let config = LangfuseConfig::from_env().unwrap();
    assert_eq!(config.host(), "https://langfuse.internal");

    std::env::remove_var(LANGFUSE_HOST);
    let config = LangfuseConfig::from_env().unwrap();
    assert_eq!(config.host(), "https://langfuse.ignored");
}

#[test]
fn from_env_names_a_missing_variable() {
    let _guard = EnvGuard::new();
    let error = LangfuseConfig::from_env().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains(LANGFUSE_PUBLIC_KEY), "{error}");

    set_env(LANGFUSE_PUBLIC_KEY, "pk-lf-public");
    let error = LangfuseConfig::from_env().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains(LANGFUSE_SECRET_KEY), "{error}");
}

#[cfg(feature = "otel")]
#[test]
fn exporter_posts_to_the_traces_signal_path() {
    let exporter = LangfuseConfig::new("pk-lf-public", "sk-lf-secret")
        .with_host("https://langfuse.internal")
        .exporter()
        .unwrap();
    assert_eq!(
        exporter.traces_url(),
        "https://langfuse.internal/api/public/otel/v1/traces",
    );
    assert!(exporter.traces_url().ends_with("/v1/traces"));
}
