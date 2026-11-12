//! Tracer handle and the no-op seam used when the feature is off.
//!
//! Port of `init_otel_tracer` from
//! `openjiuwen/extensions/tracer_otel/setup.py` (v0.1.20). Python returns a
//! `Tracer` bound to a private `TracerProvider` and carefully avoids
//! `trace.set_tracer_provider()`; Rust returns a [`TracerHandle`] that owns the
//! same private provider, so several handles can coexist (tests, agents, and
//! the observability stack) without touching process-global state.
//!
//! Two configurations share one API:
//!
//! - without `feature = "otel"`, [`init_otel_tracer`] validates the
//!   configuration and returns an inactive handle for *every*
//!   [`ExporterType`]; nothing is built and nothing fails;
//! - with `feature = "otel"`, `Memory`/`Console`/`None` build a real private
//!   provider and `Otlp` answers `Error::Unsupported` because OTLP export is
//!   deferred.
//!
//! [`ExporterType`]: super::config::ExporterType
//! [`init_otel_tracer`]: self::init_otel_tracer

#[cfg(feature = "otel")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "otel")]
use std::sync::Arc;

use openjiuwen_common::error::Result;

#[cfg(feature = "otel")]
use opentelemetry::trace::TracerProvider as _;
#[cfg(feature = "otel")]
use opentelemetry_sdk::trace::{SdkTracer, SdkTracerProvider, SpanExporter};

#[cfg(feature = "otel")]
use super::config::ExporterType;
use super::config::OtelTracerConfig;
#[cfg(feature = "otel")]
use super::provider::{
    build_provider, build_provider_without_exporter, CaptureExporter, ConsoleExporter,
};

/// Message returned when the deferred OTLP exporter is requested.
#[cfg(feature = "otel")]
const OTLP_DEFERRED: &str =
    "OTLP span export is deferred; use ExporterType::Memory/Console or inject a SpanExporter";

/// `tracing` target for handle-level failures.
#[cfg(feature = "otel")]
const LOG_TARGET: &str = "openjiuwen_extensions::otel::handle";

/// A private tracer provider plus the tracer and exporters built from one
/// [`OtelTracerConfig`].
///
/// The handle owns everything: dropping it drops the provider, and
/// [`TracerHandle::shutdown`] can be called explicitly (it is idempotent).
#[derive(Debug)]
pub struct TracerHandle {
    /// Configuration this handle was built from.
    config: OtelTracerConfig,
    /// Private provider; `None` when the handle is inactive.
    #[cfg(feature = "otel")]
    provider: Option<SdkTracerProvider>,
    /// Tracer bound to `provider`.
    #[cfg(feature = "otel")]
    tracer: Option<SdkTracer>,
    /// Exporter snapshot, present only for [`ExporterType::Memory`].
    #[cfg(feature = "otel")]
    capture: Option<Arc<CaptureExporter>>,
    /// Set by the first [`TracerHandle::shutdown`] call.
    #[cfg(feature = "otel")]
    shutdown: AtomicBool,
}

impl TracerHandle {
    /// Reports whether this handle owns a live provider.
    ///
    /// Always `false` without `feature = "otel"`, and `false` again after
    /// [`Self::shutdown`].
    #[must_use]
    pub fn is_active(&self) -> bool {
        #[cfg(feature = "otel")]
        {
            self.provider.is_some() && !self.shutdown.load(Ordering::Acquire)
        }
        #[cfg(not(feature = "otel"))]
        {
            false
        }
    }

    /// Returns the instrumentation-scope name reported with every span.
    #[must_use]
    pub fn tracer_name(&self) -> &str {
        &self.config.tracer_name
    }

    /// Returns the configuration this handle was built from.
    #[must_use]
    pub fn config(&self) -> &OtelTracerConfig {
        &self.config
    }

    /// Flushes pending spans; a no-op for an inactive handle.
    pub fn force_flush(&self) {
        #[cfg(feature = "otel")]
        {
            let Some(provider) = self.provider.as_ref().filter(|_| !self.is_shutdown()) else {
                return;
            };
            if let Err(error) = provider.force_flush() {
                tracing::warn!(
                    target: LOG_TARGET,
                    %error,
                    "failed to flush the otel tracer provider"
                );
            }
        }
    }

    /// Shuts the private provider down; safe to call twice and from `&self`.
    ///
    /// A no-op for an inactive handle.
    pub fn shutdown(&self) {
        #[cfg(feature = "otel")]
        {
            if self.is_shutdown() || self.shutdown.swap(true, Ordering::AcqRel) {
                return;
            }
            if let Some(provider) = self.provider.as_ref() {
                if let Err(error) = provider.shutdown() {
                    tracing::warn!(
                        target: LOG_TARGET,
                        %error,
                        "failed to shut the otel tracer provider down"
                    );
                }
            }
        }
    }

    /// Returns a clone of the tracer bound to the private provider.
    ///
    /// `None` for an inactive handle.
    #[cfg(feature = "otel")]
    #[must_use]
    pub fn tracer(&self) -> Option<SdkTracer> {
        self.tracer.clone()
    }

    /// Returns the private provider.
    ///
    /// `None` for an inactive handle.
    #[cfg(feature = "otel")]
    #[must_use]
    pub fn provider(&self) -> Option<&SdkTracerProvider> {
        self.provider.as_ref()
    }

    /// Returns the capturing exporter, present only when the configuration
    /// asked for [`ExporterType::Memory`].
    #[cfg(feature = "otel")]
    #[must_use]
    pub fn capture(&self) -> Option<Arc<CaptureExporter>> {
        self.capture.clone()
    }

    /// Reports whether [`Self::shutdown`] already ran (feature builds only).
    #[cfg(feature = "otel")]
    fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Acquire)
    }

    /// Builds a handle without a provider (used by the no-feature seam).
    #[cfg(not(feature = "otel"))]
    fn inactive(config: OtelTracerConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "otel")]
            provider: None,
            #[cfg(feature = "otel")]
            tracer: None,
            #[cfg(feature = "otel")]
            capture: None,
            #[cfg(feature = "otel")]
            shutdown: AtomicBool::new(false),
        }
    }

    /// Builds a handle around a live provider.
    #[cfg(feature = "otel")]
    fn active(
        config: OtelTracerConfig,
        provider: SdkTracerProvider,
        tracer: SdkTracer,
        capture: Option<Arc<CaptureExporter>>,
    ) -> Self {
        Self {
            config,
            provider: Some(provider),
            tracer: Some(tracer),
            capture,
            shutdown: AtomicBool::new(false),
        }
    }
}

/// Initializes the tracer, without the `otel` feature.
///
/// The configuration is validated and kept in the returned handle, but no
/// provider is built: [`TracerHandle::is_active`] is `false` for every
/// [`ExporterType`](super::config::ExporterType).
///
/// # Errors
///
/// Returns [`openjiuwen_common::error::Error::validation`] when the
/// configuration is invalid (for example a `sample_rate` outside `0.0..=1.0`).
#[cfg(not(feature = "otel"))]
pub fn init_otel_tracer(config: OtelTracerConfig) -> Result<TracerHandle> {
    config.validate()?;
    Ok(TracerHandle::inactive(config))
}

/// Initializes the tracer and returns its private provider handle.
///
/// `Memory` attaches a [`CaptureExporter`], `Console` exports one `tracing`
/// line per span at `debug`, and `None` builds a provider whose spans are
/// dropped. OTLP export is deferred and answers
/// [`openjiuwen_common::error::Error::unsupported`].
///
/// # Errors
///
/// Returns [`openjiuwen_common::error::Error::validation`] when the
/// configuration is invalid, and
/// [`openjiuwen_common::error::Error::Unsupported`] for
/// [`ExporterType::Otlp`].
#[cfg(feature = "otel")]
pub fn init_otel_tracer(config: OtelTracerConfig) -> Result<TracerHandle> {
    config.validate()?;
    match config.exporter_type {
        ExporterType::Otlp => Err(otlp_deferred()),
        ExporterType::Memory => {
            let capture = Arc::new(CaptureExporter::new());
            let provider = build_provider(&config, capture.as_ref().clone())?;
            let tracer = provider.tracer(config.tracer_name.clone());
            Ok(TracerHandle::active(
                config,
                provider,
                tracer,
                Some(capture),
            ))
        }
        ExporterType::Console => {
            let provider = build_provider(&config, ConsoleExporter)?;
            let tracer = provider.tracer(config.tracer_name.clone());
            Ok(TracerHandle::active(config, provider, tracer, None))
        }
        ExporterType::None => {
            let provider = build_provider_without_exporter(&config)?;
            let tracer = provider.tracer(config.tracer_name.clone());
            Ok(TracerHandle::active(config, provider, tracer, None))
        }
    }
}

/// Initializes the tracer around a caller-supplied [`SpanExporter`].
///
/// Useful for hosts that already own an exporter (OTLP, a test sink, a vendor
/// bridge) and for tests that need to observe spans directly. The provider is
/// still private to the returned handle and [`TracerHandle::capture`] stays
/// `None`, because the handle cannot see inside a foreign exporter.
///
/// # Errors
///
/// Returns [`openjiuwen_common::error::Error::validation`] when the
/// configuration is invalid, and
/// [`openjiuwen_common::error::Error::Unsupported`] when the configuration
/// still asks for the deferred [`ExporterType::Otlp`].
#[cfg(feature = "otel")]
pub fn init_otel_tracer_with_exporter<E>(
    config: OtelTracerConfig,
    exporter: E,
) -> Result<TracerHandle>
where
    E: SpanExporter + 'static,
{
    config.validate()?;
    if config.exporter_type == ExporterType::Otlp {
        return Err(otlp_deferred());
    }
    let provider = build_provider(&config, exporter)?;
    let tracer = provider.tracer(config.tracer_name.clone());
    Ok(TracerHandle::active(config, provider, tracer, None))
}

/// The error returned for the deferred OTLP exporter.
#[cfg(feature = "otel")]
fn otlp_deferred() -> openjiuwen_common::error::Error {
    openjiuwen_common::error::Error::unsupported(OTLP_DEFERRED)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::otel::config::ExporterType;

    // port of tests/unit_tests/extensions/tracer_otel/test_setup.py

    /// Configuration that selects a capturing provider.
    fn memory_config() -> OtelTracerConfig {
        OtelTracerConfig::default().with_exporter_type(ExporterType::Memory)
    }

    #[test]
    fn memory_exporter_matches_the_feature_configuration() {
        let handle = init_otel_tracer(memory_config()).unwrap();
        assert_eq!(handle.is_active(), cfg!(feature = "otel"));
        assert_eq!(handle.tracer_name(), "openjiuwen.tracer.otel");
        assert_eq!(handle.config().exporter_type, ExporterType::Memory);
    }

    #[test]
    fn every_exporter_initializes_without_the_feature() {
        for exporter_type in [
            ExporterType::None,
            ExporterType::Memory,
            ExporterType::Console,
            ExporterType::Otlp,
        ] {
            let config = OtelTracerConfig::default().with_exporter_type(exporter_type);
            let result = init_otel_tracer(config);
            if cfg!(feature = "otel") && exporter_type == ExporterType::Otlp {
                assert!(result.is_err(), "OTLP must stay deferred");
            } else {
                let handle = result.unwrap();
                assert_eq!(handle.is_active(), cfg!(feature = "otel"));
                assert_eq!(handle.config().exporter_type, exporter_type);
            }
        }
    }

    #[test]
    fn tracer_name_and_config_are_preserved() {
        let config = OtelTracerConfig {
            tracer_name: "custom.tracer".to_owned(),
            sample_rate: 0.25,
            ..memory_config()
        };
        let handle = init_otel_tracer(config.clone()).unwrap();
        assert_eq!(handle.tracer_name(), "custom.tracer");
        assert_eq!(handle.config(), &config);
    }

    #[test]
    fn invalid_sample_rate_is_rejected_before_the_provider_is_built() {
        let config = OtelTracerConfig {
            sample_rate: 1.5,
            ..memory_config()
        };
        let error = init_otel_tracer(config).unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert!(error.to_string().contains("sample_rate must be between"));
    }

    #[test]
    fn shutdown_and_force_flush_are_safe_to_repeat() {
        let handle = init_otel_tracer(memory_config()).unwrap();
        handle.force_flush();
        handle.force_flush();
        handle.shutdown();
        handle.shutdown();
        assert!(!handle.is_active());
        // Post-shutdown calls stay harmless.
        handle.force_flush();
        handle.shutdown();
    }

    #[cfg(feature = "otel")]
    #[test]
    fn otlp_exporter_is_deferred() {
        let config = OtelTracerConfig::default().with_exporter_type(ExporterType::Otlp);
        let error = init_otel_tracer(config).unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Unsupported
        );
        assert_eq!(
            error.to_string(),
            "unsupported: OTLP span export is deferred; use ExporterType::Memory/Console or \
             inject a SpanExporter"
        );
    }

    #[cfg(feature = "otel")]
    #[test]
    fn memory_exporter_captures_while_console_and_none_do_not() {
        let memory = init_otel_tracer(memory_config()).unwrap();
        assert!(memory.is_active());
        assert!(memory.capture().is_some());
        assert!(memory.provider().is_some());

        for exporter_type in [ExporterType::Console, ExporterType::None] {
            let config = OtelTracerConfig::default().with_exporter_type(exporter_type);
            let handle = init_otel_tracer(config).unwrap();
            assert!(handle.is_active(), "{exporter_type:?} must be active");
            assert!(handle.capture().is_none());
        }
    }

    #[cfg(feature = "otel")]
    #[test]
    fn injected_exporter_stays_outside_the_capture_snapshot() {
        use crate::otel::provider::CaptureExporter;
        use opentelemetry::trace::{Span as _, Tracer as _};

        let capture = CaptureExporter::new();
        let handle = init_otel_tracer_with_exporter(memory_config(), capture.clone()).unwrap();
        assert!(handle.is_active());
        assert!(handle.capture().is_none());

        handle.tracer().unwrap().start("injected.span").end();
        handle.force_flush();
        assert_eq!(capture.names(), vec!["injected.span".to_owned()]);
    }

    #[cfg(feature = "otel")]
    #[test]
    fn injected_exporter_still_rejects_otlp() {
        use crate::otel::provider::CaptureExporter;

        let config = OtelTracerConfig::default().with_exporter_type(ExporterType::Otlp);
        let error = init_otel_tracer_with_exporter(config, CaptureExporter::new()).unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Unsupported
        );
    }
}
