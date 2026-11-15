//! Crate-boundary tests of the `otel` seam that build with and without the
//! feature.
//!
//! Python counterpart: `tests/unit_tests/extensions/tracer_otel/test_setup.py`
//! (the exporter-type handling of `OtelTracer.setup`). Without `feature =
//! "otel"` `init_otel_tracer` validates the configuration and then splits by
//! exporter type: `Memory`, `Console` and `None` return an inactive handle (the
//! no-op seam), while `Otlp` and `File` answer `Error::Unsupported` naming the
//! missing `otel` cargo feature. With the feature every exporter type builds a
//! private provider, and `ExporterType::Otlp` posts OTLP/HTTP batches in the
//! JSON (`http/json`) or protobuf (`http/protobuf`) encoding, so only the
//! unsupported `OtlpProtocol::Grpc` transport is rejected.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::Duration;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_extensions::otel::{init_otel_tracer, ExporterType, OtelTracerConfig, OtlpProtocol};

/// Every exporter type a feature build turns into a real provider, in
/// declaration order.
///
/// `ExporterType::File` is covered by
/// `the_file_exporter_initializes_into_a_traces_directory`, which needs its own
/// temporary traces directory.
#[cfg(feature = "otel")]
const EXPORTER_TYPES: [ExporterType; 4] = [
    ExporterType::None,
    ExporterType::Memory,
    ExporterType::Console,
    ExporterType::Otlp,
];

/// A configuration selecting `exporter_type` with the supported OTLP transport.
///
/// The protocol and the endpoint are explicit, so no test depends on the
/// ambient `OPENJIUWEN_OTEL_*`/`OTEL_*` variables, and the address stays on
/// loopback. No test creates a span for `Otlp`, so nothing is ever posted.
fn config(exporter_type: ExporterType) -> OtelTracerConfig {
    config_with_protocol(exporter_type, OtlpProtocol::Http)
}

/// The same offline configuration, selecting `protocol` explicitly.
///
/// [`config`] keeps selecting the JSON encoding; this helper adds the protobuf
/// one without duplicating the endpoint and timeout defaults.
fn config_with_protocol(exporter_type: ExporterType, protocol: OtlpProtocol) -> OtelTracerConfig {
    OtelTracerConfig {
        protocol,
        exporter_endpoint: Some("http://127.0.0.1:9".to_owned()),
        export_timeout_ms: 500,
        ..OtelTracerConfig::default()
    }
    .with_exporter_type(exporter_type)
}

#[cfg(feature = "otel")]
#[test]
fn every_exporter_type_initializes() {
    for exporter_type in EXPORTER_TYPES {
        let handle = init_otel_tracer(config(exporter_type))
            .unwrap_or_else(|error| panic!("{exporter_type:?} must initialize: {error}"));
        assert_eq!(handle.tracer_name(), "openjiuwen.tracer.otel");
        assert_eq!(handle.config().exporter_type, exporter_type);
        // A feature build owns a private provider for every exporter type.
        assert!(handle.is_active(), "{exporter_type:?} must be active");
        handle.force_flush();
        handle.shutdown();
        assert!(!handle.is_active(), "shutdown deactivates the handle");
        handle.shutdown(); // idempotent
        handle.shutdown_with_timeout(Duration::from_millis(50)); // idempotent
        handle.force_flush(); // harmless after shutdown
    }
}

#[cfg(not(feature = "otel"))]
#[test]
fn memory_console_and_none_stay_an_inactive_seam() {
    // The three exporter types that need no SDK stay a documented no-op seam:
    // the handle keeps the configuration and `is_active()` is the signal.
    for exporter_type in [
        ExporterType::None,
        ExporterType::Memory,
        ExporterType::Console,
    ] {
        let handle = init_otel_tracer(config(exporter_type))
            .unwrap_or_else(|error| panic!("{exporter_type:?} must initialize: {error}"));
        assert_eq!(handle.tracer_name(), "openjiuwen.tracer.otel");
        assert_eq!(handle.config().exporter_type, exporter_type);
        assert!(
            !handle.is_active(),
            "{exporter_type:?} owns no provider without the feature"
        );
        handle.force_flush();
        handle.shutdown();
        assert!(!handle.is_active(), "shutdown stays a no-op");
        handle.shutdown(); // idempotent
        handle.shutdown_with_timeout(Duration::from_millis(50)); // idempotent
        handle.force_flush(); // harmless after shutdown
    }
}

#[cfg(not(feature = "otel"))]
#[test]
fn otlp_and_file_are_unsupported_without_the_feature() {
    // A collector endpoint and a traces directory promise exported telemetry;
    // without the SDK nothing can honour them, so both answer a typed error
    // naming the `otel` cargo feature rather than an inactive handle.
    let dir = tempfile::tempdir().expect("a temporary traces directory");
    let file_config = OtelTracerConfig {
        traces_dir: dir.path().to_string_lossy().into_owned(),
        file_retention_days: 0,
        ..config(ExporterType::File)
    };
    for (exporter_type, config) in [
        (ExporterType::Otlp, config(ExporterType::Otlp)),
        (ExporterType::File, file_config),
    ] {
        let error = init_otel_tracer(config)
            .expect_err("an SDK-only exporter must not initialize without the feature");
        assert_eq!(error.kind(), ErrorKind::Unsupported, "{exporter_type:?}");
        let message = error.to_string();
        assert!(
            message.contains("otel"),
            "the reason must name the cargo feature: {message}"
        );
        assert!(
            message.contains("cargo feature off"),
            "the reason must say the feature is off: {message}"
        );
        assert!(
            message.contains("enable the `otel` feature"),
            "the reason must name the fix: {message}"
        );
    }
    assert!(
        dir.path()
            .read_dir()
            .expect("a readable directory")
            .next()
            .is_none(),
        "a rejected File configuration must not create or write trace files"
    );
}

#[cfg(not(feature = "otel"))]
#[test]
fn a_configuration_error_precedes_the_missing_feature() {
    // Validation runs first, so `Otlp` reports the configuration error in this
    // build too instead of `Unsupported`.
    let invalid = OtelTracerConfig {
        sample_rate: 1.5,
        ..config(ExporterType::Otlp)
    };
    let error = init_otel_tracer(invalid).expect_err("sample_rate must stay within 0..=1");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("sample_rate must be between"));
}

#[test]
fn a_configuration_error_is_reported_in_both_builds() {
    // test_setup.py: sample_rate outside 0.0..=1.0 is rejected before any
    // provider is built, with or without the feature.
    let invalid = OtelTracerConfig {
        sample_rate: 1.5,
        ..config(ExporterType::Memory)
    };
    let error = init_otel_tracer(invalid).expect_err("sample_rate must stay within 0..=1");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("sample_rate must be between"));
}

#[cfg(feature = "otel")]
#[test]
fn the_memory_exporter_owns_a_provider_that_captures() {
    let handle = init_otel_tracer(config(ExporterType::Memory))
        .expect("the memory exporter always initializes");
    assert!(handle.is_active());
    assert!(
        handle.tracer().is_some(),
        "an active handle exposes a tracer"
    );
    assert!(
        handle.capture().is_some(),
        "the memory exporter attaches the capture API"
    );

    for exporter_type in [ExporterType::Console, ExporterType::None] {
        let handle = init_otel_tracer(config(exporter_type))
            .expect("the console and no-op exporters always initialize");
        assert!(handle.is_active(), "{exporter_type:?} must be active");
        assert!(
            handle.capture().is_none(),
            "{exporter_type:?} exposes no capture API"
        );
    }
}

#[cfg(feature = "otel")]
#[test]
fn the_otlp_http_exporter_owns_a_provider_without_a_capture_snapshot() {
    let handle = init_otel_tracer(config(ExporterType::Otlp))
        .expect("the OTLP/HTTP exporter always initializes");
    assert!(handle.is_active());
    assert!(handle.tracer().is_some());
    assert!(
        handle.capture().is_none(),
        "the OTLP exporter exposes no capture API"
    );

    handle.force_flush();
    handle.shutdown();
    assert!(!handle.is_active(), "shutdown deactivates the handle");
    handle.shutdown(); // idempotent after the first call
}

#[cfg(feature = "otel")]
#[test]
fn the_otlp_http_protobuf_exporter_owns_a_provider_without_a_capture_snapshot() {
    let handle = init_otel_tracer(config_with_protocol(
        ExporterType::Otlp,
        OtlpProtocol::HttpProtobuf,
    ))
    .expect("the OTLP/HTTP protobuf exporter always initializes");
    assert!(handle.is_active());
    assert_eq!(handle.config().protocol, OtlpProtocol::HttpProtobuf);
    assert!(handle.tracer().is_some());
    assert!(
        handle.capture().is_none(),
        "the OTLP exporter exposes no capture API"
    );

    // Nothing is exported, so the closed loopback endpoint is never contacted.
    handle.force_flush();
    handle.shutdown();
    assert!(!handle.is_active(), "shutdown deactivates the handle");
    handle.shutdown(); // idempotent after the first call
    handle.shutdown_with_timeout(Duration::from_millis(50)); // idempotent
    handle.force_flush(); // harmless after shutdown
}

#[cfg(feature = "otel")]
#[test]
fn the_default_otlp_grpc_combination_is_unsupported() {
    // `OtlpProtocol::Grpc` is the configuration default, but an ambient
    // protocol variable would override it; clear both so the default decides.
    // No other test in this binary depends on those variables: every OTLP
    // handle here sets the protocol explicitly.
    for name in [
        "OPENJIUWEN_OTEL_EXPORTER_OTLP_PROTOCOL",
        "OTEL_EXPORTER_OTLP_PROTOCOL",
    ] {
        std::env::remove_var(name);
    }

    let config = OtelTracerConfig::default().with_exporter_type(ExporterType::Otlp);
    assert_eq!(config.protocol, OtlpProtocol::Grpc);

    let error = init_otel_tracer(config).expect_err("OTLP/gRPC is not implemented");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    let message = error.to_string();
    assert!(message.contains("gRPC"), "unexpected message: {message}");
    assert!(
        message.contains("gRPC framing"),
        "the reason is the gRPC framing, not a missing encoding: {message}"
    );
    assert!(
        message.contains("http/json"),
        "the reason must name the JSON transport: {message}"
    );
    assert!(
        message.contains("http/protobuf"),
        "the reason must name the protobuf transport: {message}"
    );
}

#[cfg(feature = "otel")]
#[test]
fn the_file_exporter_initializes_into_a_traces_directory() {
    // `ExporterType::File` needs no collector; the directory stays inside a
    // temporary dir so the test never writes into the workspace.
    let dir = tempfile::tempdir().expect("a temporary traces directory");
    let config = OtelTracerConfig {
        traces_dir: dir.path().to_string_lossy().into_owned(),
        file_retention_days: 0,
        ..config(ExporterType::File)
    };
    let handle = init_otel_tracer(config).expect("the file exporter always initializes");
    assert!(handle.is_active());
    assert_eq!(handle.config().exporter_type, ExporterType::File);
    assert!(handle.tracer().is_some());
    assert!(
        handle.capture().is_none(),
        "the file exporter exposes no capture API"
    );

    handle.force_flush();
    handle.shutdown();
    assert!(!handle.is_active(), "shutdown deactivates the handle");
    handle.shutdown(); // idempotent after the first call
}
