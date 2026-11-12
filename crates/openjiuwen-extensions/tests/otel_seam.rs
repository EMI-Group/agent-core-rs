//! Crate-boundary tests of the `otel` seam that build with and without the
//! feature.
//!
//! Python counterpart: `tests/unit_tests/extensions/tracer_otel/test_setup.py`
//! (the exporter-type handling of `OtelTracer.setup`). Without `feature =
//! "otel"` `init_otel_tracer` validates the configuration and returns an
//! inactive handle for every `ExporterType`; with the feature the in-memory,
//! console, and no-op exporters build a private provider while OTLP export
//! stays deferred.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use openjiuwen_common::error::ErrorKind;
use openjiuwen_extensions::otel::{init_otel_tracer, ExporterType, OtelTracerConfig};

/// Every exporter type the configuration offers, in declaration order.
const EXPORTER_TYPES: [ExporterType; 4] = [
    ExporterType::None,
    ExporterType::Memory,
    ExporterType::Console,
    ExporterType::Otlp,
];

/// A default configuration selecting `exporter_type`.
fn config(exporter_type: ExporterType) -> OtelTracerConfig {
    OtelTracerConfig::default().with_exporter_type(exporter_type)
}

#[test]
fn every_exporter_type_initializes() {
    for exporter_type in EXPORTER_TYPES {
        // OTLP is the only deferred exporter, and only in a feature build.
        if cfg!(feature = "otel") && exporter_type == ExporterType::Otlp {
            let error = init_otel_tracer(config(exporter_type)).expect_err("OTLP stays deferred");
            assert_eq!(error.kind(), ErrorKind::Unsupported);
            assert!(error.to_string().contains("OTLP span export is deferred"));
            continue;
        }

        let handle =
            init_otel_tracer(config(exporter_type)).expect("every other exporter type initializes");
        assert_eq!(handle.tracer_name(), "openjiuwen.tracer.otel");
        assert_eq!(handle.config().exporter_type, exporter_type);
        // Only a feature build owns a provider; the no-feature seam is inert.
        assert_eq!(handle.is_active(), cfg!(feature = "otel"));
        handle.force_flush();
        handle.shutdown();
        assert!(!handle.is_active(), "shutdown deactivates the handle");
        handle.shutdown(); // idempotent
        handle.force_flush(); // harmless after shutdown
    }
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
fn otlp_is_deferred_with_the_feature() {
    let error = init_otel_tracer(config(ExporterType::Otlp)).expect_err("OTLP export is deferred");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.to_string().contains("OTLP span export is deferred"));
}
