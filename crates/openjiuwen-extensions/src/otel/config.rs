//! Tracer configuration: exporter, sampling, redaction, and batch export knobs.
//!
//! Port of `openjiuwen/extensions/tracer_otel/config.py` (v0.1.20). The Python
//! class is a frozen dataclass whose `__post_init__` validates the sampling
//! probability; Rust keeps the same fields and defaults, turns the constructor
//! check into an explicit [`OtelTracerConfig::validate`], and derives the
//! builder methods that a frozen dataclass would spell with `dataclasses.replace`.
//!
//! This module compiles with and without the `otel` feature: it holds no SDK
//! types, so hosts can build and validate configuration without the SDK.

use std::collections::BTreeMap;

use openjiuwen_common::config::env::{flag, OPENJIUWEN_OTEL_GLOBAL_INSTRUMENT_ENABLE};
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Default tracer instrumentation name: `"openjiuwen.tracer.otel"`.
const DEFAULT_TRACER_NAME: &str = "openjiuwen.tracer.otel";
/// Default `service.name` resource attribute: `"openjiuwen"`.
const DEFAULT_SERVICE_NAME: &str = "openjiuwen";
/// Lowest accepted sampling probability (inclusive).
const MIN_SAMPLE_RATE: f64 = 0.0;
/// Highest accepted sampling probability (inclusive).
const MAX_SAMPLE_RATE: f64 = 1.0;

/// Which span exporter the tracer provider should install.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExporterType {
    /// Build a provider that records nothing; the tracer is a no-op.
    None,
    /// Keep finished spans in memory where tests can read them.
    Memory,
    /// Write one `tracing` line per finished span at `debug`.
    Console,
    /// Ship spans to an OTLP collector (deferred; the surface is stable).
    #[default]
    Otlp,
}

/// Transport protocol used by the (deferred) OTLP exporter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OtlpProtocol {
    /// OTLP over gRPC.
    #[default]
    Grpc,
    /// OTLP over HTTP with protobuf payloads.
    Http,
}

/// Immutable configuration for the `OTel` tracer extension.
///
/// Independent from any `ObservabilityConfig`: `tracer_otel` lives in extensions
/// and must not depend on the agent-team layer (Python parity).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct OtelTracerConfig {
    /// Instrumentation-scope name reported with every span.
    pub tracer_name: String,
    /// Which exporter the provider should install.
    pub exporter_type: ExporterType,
    /// Collector endpoint for the OTLP exporter; `None` keeps the SDK default.
    pub exporter_endpoint: Option<String>,
    /// OTLP transport protocol.
    pub protocol: OtlpProtocol,
    /// Extra headers sent with every OTLP request (authorization, tenant, …).
    pub headers: BTreeMap<String, String>,
    /// `service.name` resource attribute.
    pub service_name: String,
    /// `service.version` resource attribute; `"unknown"` is applied at provider
    /// build time when this stays `None`.
    pub service_version: Option<String>,
    /// Trace sampling probability, within `0.0..=1.0`.
    pub sample_rate: f64,
    /// Batch span processor export interval in milliseconds.
    pub schedule_delay_millis: u64,
    /// Batch span processor export timeout in milliseconds.
    pub export_timeout_ms: u64,
    /// Maximum number of spans in one export batch.
    pub max_export_batch_size: usize,
    /// Legacy switch: hash prompt and completion text when redaction is on.
    pub redaction_enabled: bool,
    /// Per-field prompt override; `None` follows [`Self::redaction_enabled`].
    pub redact_prompts: Option<bool>,
    /// Per-field completion override; `None` follows [`Self::redaction_enabled`].
    pub redact_completions: Option<bool>,
    /// General attribute cap; canonical system instructions are exempt.
    pub max_attr_length: usize,
    /// Span attribute count limit.
    pub max_attributes: usize,
    /// Process-wide HTTP instrumentation switch (traceparent propagation).
    ///
    /// [`global_instrument_enabled`] lets the
    /// `OPENJIUWEN_OTEL_GLOBAL_INSTRUMENT_ENABLE` variable override this flag
    /// without affecting the private tracer provider.
    pub global_instrument_enable: bool,
}

impl Default for OtelTracerConfig {
    fn default() -> Self {
        Self {
            tracer_name: DEFAULT_TRACER_NAME.to_owned(),
            exporter_type: ExporterType::default(),
            exporter_endpoint: None,
            protocol: OtlpProtocol::default(),
            headers: BTreeMap::new(),
            service_name: DEFAULT_SERVICE_NAME.to_owned(),
            service_version: None,
            sample_rate: 1.0,
            schedule_delay_millis: 5_000,
            export_timeout_ms: 30_000,
            max_export_batch_size: 512,
            redaction_enabled: true,
            redact_prompts: None,
            redact_completions: None,
            max_attr_length: 4_096,
            max_attributes: 200,
            global_instrument_enable: false,
        }
    }
}

impl OtelTracerConfig {
    /// Checks the configuration before a provider is built.
    ///
    /// Mirrors the Python `__post_init__` range check and is therefore also the
    /// validation step of the no-`otel`-feature seam.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] ([`openjiuwen_common::error::ErrorKind::Validation`])
    /// when [`Self::sample_rate`] is outside `0.0..=1.0`; a `NaN` rate is
    /// rejected because it never compares inside the range.
    pub fn validate(&self) -> Result<()> {
        if !(MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&self.sample_rate) {
            return Err(Error::validation(format!(
                "sample_rate must be between 0.0 and 1.0, got {}",
                self.sample_rate
            )));
        }
        Ok(())
    }

    /// Returns the configuration with a different [`ExporterType`].
    #[must_use]
    pub fn with_exporter_type(mut self, exporter_type: ExporterType) -> Self {
        self.exporter_type = exporter_type;
        self
    }

    /// Returns the configuration with an OTLP collector endpoint.
    #[must_use]
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.exporter_endpoint = Some(endpoint.into());
        self
    }

    /// Returns the configuration with a different `service.name` and
    /// `service.version`.
    #[must_use]
    pub fn with_service(mut self, name: impl Into<String>, version: Option<String>) -> Self {
        self.service_name = name.into();
        self.service_version = version;
        self
    }
}

/// Reports whether process-wide HTTP instrumentation is enabled.
///
/// `OPENJIUWEN_OTEL_GLOBAL_INSTRUMENT_ENABLE` wins over
/// [`OtelTracerConfig::global_instrument_enable`] when it is set and non-blank
/// (`"1"`, `"true"`, `"yes"`, `"on"` in any case are true); otherwise the
/// struct flag is returned unchanged.
#[must_use]
pub fn global_instrument_enabled(config: &OtelTracerConfig) -> bool {
    flag(OPENJIUWEN_OTEL_GLOBAL_INSTRUMENT_ENABLE).unwrap_or(config.global_instrument_enable)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::{json, Value};

    // port of tests/unit_tests/extensions/tracer_otel/test_setup.py (config cases)

    #[test]
    fn defaults_match_python_dataclass() {
        let config = OtelTracerConfig::default();
        assert_eq!(config.tracer_name, "openjiuwen.tracer.otel");
        assert_eq!(config.exporter_type, ExporterType::Otlp);
        assert_eq!(config.exporter_endpoint, None);
        assert_eq!(config.protocol, OtlpProtocol::Grpc);
        assert!(config.headers.is_empty());
        assert_eq!(config.service_name, "openjiuwen");
        assert_eq!(config.service_version, None);
        assert!((config.sample_rate - 1.0).abs() < f64::EPSILON);
        assert_eq!(config.schedule_delay_millis, 5_000);
        assert_eq!(config.export_timeout_ms, 30_000);
        assert_eq!(config.max_export_batch_size, 512);
        assert!(config.redaction_enabled);
        assert_eq!(config.redact_prompts, None);
        assert_eq!(config.redact_completions, None);
        assert_eq!(config.max_attr_length, 4_096);
        assert_eq!(config.max_attributes, 200);
        assert!(!config.global_instrument_enable);
    }

    #[test]
    fn enum_defaults_follow_python() {
        assert_eq!(ExporterType::default(), ExporterType::Otlp);
        assert_eq!(OtlpProtocol::default(), OtlpProtocol::Grpc);
    }

    fn wire_json() -> Value {
        json!({
            "tracer_name": "custom.tracer",
            "exporter_type": "memory",
            "exporter_endpoint": "http://localhost:4318",
            "protocol": "http",
            "headers": {"authorization": "Bearer token"},
            "service_name": "custom-service",
            "service_version": "1.2.3",
            "sample_rate": 0.25,
            "schedule_delay_millis": 1000,
            "export_timeout_ms": 2000,
            "max_export_batch_size": 64,
            "redaction_enabled": false,
            "redact_prompts": true,
            "redact_completions": false,
            "max_attr_length": 128,
            "max_attributes": 42,
            "global_instrument_enable": true
        })
    }

    #[test]
    fn deserializes_snake_case_wire_keys() {
        let config: OtelTracerConfig = serde_json::from_value(wire_json()).unwrap();
        assert_eq!(config.tracer_name, "custom.tracer");
        assert_eq!(config.exporter_type, ExporterType::Memory);
        assert_eq!(config.protocol, OtlpProtocol::Http);
        assert_eq!(config.headers["authorization"], "Bearer token");
        assert_eq!(config.service_version.as_deref(), Some("1.2.3"));
        assert_eq!(config.redact_prompts, Some(true));
        assert_eq!(config.redact_completions, Some(false));
        assert_eq!(config.max_attributes, 42);
        assert!(config.global_instrument_enable);
    }

    #[test]
    fn serializes_snake_case_enum_values() {
        let value = serde_json::to_value(OtelTracerConfig::default()).unwrap();
        assert_eq!(value["exporter_type"], json!("otlp"));
        assert_eq!(value["protocol"], json!("grpc"));
        assert_eq!(value["max_export_batch_size"], json!(512));

        let memory = OtelTracerConfig::default()
            .with_exporter_type(ExporterType::Memory)
            .with_service("svc", None);
        let value = serde_json::to_value(memory).unwrap();
        assert_eq!(value["exporter_type"], json!("memory"));
        assert_eq!(value["service_name"], json!("svc"));
        assert_eq!(value["service_version"], Value::Null);
    }

    #[test]
    fn round_trips_through_json() {
        let config: OtelTracerConfig = serde_json::from_value(wire_json()).unwrap();
        let value = serde_json::to_value(&config).unwrap();
        assert_eq!(value, wire_json());
        let restored: OtelTracerConfig = serde_json::from_value(value).unwrap();
        assert_eq!(restored, config);
    }

    #[test]
    fn tolerates_unknown_fields_and_missing_fields() {
        let mut value = wire_json();
        value["future_knob"] = json!({"nested": true});
        let config: OtelTracerConfig = serde_json::from_value(value).unwrap();
        assert_eq!(config.exporter_type, ExporterType::Memory);

        let empty: OtelTracerConfig = serde_json::from_value(json!({})).unwrap();
        assert_eq!(empty, OtelTracerConfig::default());
    }

    #[test]
    fn validate_accepts_inclusive_boundaries() {
        assert!(OtelTracerConfig::default().validate().is_ok());
        for rate in [0.0, 0.5, 1.0] {
            let config = OtelTracerConfig {
                sample_rate: rate,
                ..OtelTracerConfig::default()
            };
            assert!(
                config.validate().is_ok(),
                "sample_rate {rate} must be valid"
            );
        }
    }

    #[test]
    fn validate_rejects_out_of_range_and_nan() {
        for rate in [-0.1, 1.5, f64::NEG_INFINITY, f64::INFINITY, f64::NAN] {
            let config = OtelTracerConfig {
                sample_rate: rate,
                ..OtelTracerConfig::default()
            };
            let error = config.validate().unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation);
            let message = error.to_string();
            assert!(
                message.contains("sample_rate must be between 0.0 and 1.0"),
                "unexpected message: {message}"
            );
        }
    }

    #[test]
    fn validate_message_names_the_rejected_value() {
        let config = OtelTracerConfig {
            sample_rate: 1.5,
            ..OtelTracerConfig::default()
        };
        let error = config.validate().unwrap_err();
        assert!(error.to_string().ends_with(", got 1.5"), "{error}");
    }

    #[test]
    fn builders_chain() {
        let config = OtelTracerConfig::default()
            .with_exporter_type(ExporterType::Console)
            .with_endpoint("http://collector:4317")
            .with_service("custom-service", Some("2.0.0".to_owned()));
        assert_eq!(config.exporter_type, ExporterType::Console);
        assert_eq!(
            config.exporter_endpoint.as_deref(),
            Some("http://collector:4317")
        );
        assert_eq!(config.service_name, "custom-service");
        assert_eq!(config.service_version.as_deref(), Some("2.0.0"));

        let cleared = config.with_service("other-service", None);
        assert_eq!(cleared.service_name, "other-service");
        assert_eq!(cleared.service_version, None);
    }

    #[test]
    fn endpoint_builder_accepts_string_and_str() {
        let from_str = OtelTracerConfig::default().with_endpoint("http://a");
        let from_string = OtelTracerConfig::default().with_endpoint(String::from("http://a"));
        assert_eq!(from_str, from_string);
    }

    #[test]
    fn global_instrument_falls_back_to_struct_flag_when_env_unset() {
        // `set_var`/`remove_var` are unsafe in this edition, so a test may not
        // force the environment; when the variable happens to be set, return.
        if std::env::var(OPENJIUWEN_OTEL_GLOBAL_INSTRUMENT_ENABLE).is_ok() {
            return;
        }
        assert!(!global_instrument_enabled(&OtelTracerConfig::default()));
        let enabled = OtelTracerConfig {
            global_instrument_enable: true,
            ..OtelTracerConfig::default()
        };
        assert!(global_instrument_enabled(&enabled));
    }
}
