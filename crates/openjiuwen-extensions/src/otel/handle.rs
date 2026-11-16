//! Tracer handle and the no-op seam used when the feature is off.
//!
//! Port of `init_otel_tracer` from
//! `openjiuwen/extensions/tracer_otel/setup.py` (v0.1.20). Python returns a
//! `Tracer` bound to a private `TracerProvider` and carefully avoids
//! `trace.set_tracer_provider()`; Rust returns a [`TracerHandle`] that owns the
//! same private provider, so several handles can coexist (tests, agents, and
//! the observability stack) without touching process-global state.
//!
//! Two configurations share one API, and the no-feature build is deliberately
//! asymmetric at the exporter-type level:
//!
//! - without `feature = "otel"`, [`init_otel_tracer`] validates the
//!   configuration first and then splits by exporter type.
//!   [`ExporterType::Memory`], [`ExporterType::Console`] and
//!   [`ExporterType::None`] return an inactive handle: that is the documented
//!   no-op seam, and [`TracerHandle::is_active`] returning `false` is its
//!   signal. [`ExporterType::Otlp`] and [`ExporterType::File`] answer
//!   [the unsupported error](openjiuwen_common::error::Error::Unsupported)
//!   instead, naming the missing `otel` cargo feature: a collector endpoint or
//!   a traces directory is a request to export telemetry, and this build has
//!   no provider and no exporter to serve it, so those two never degrade
//!   silently into an inactive handle;
//! - with `feature = "otel"`, every [`ExporterType`] builds a real private
//!   provider: `Memory` keeps finished spans in memory, `Console` logs one
//!   `tracing` line per span, `File` appends single-span OTLP JSON lines to a
//!   daily trace file (no collector), `None` records nothing, and `Otlp` posts
//!   batches to a collector over OTLP/HTTP as JSON or protobuf payloads.
//!
//! # OTLP transport
//!
//! `Otlp` first resolves the transport with
//! [`env::resolve_protocol`](super::env::resolve_protocol): both
//! [`OtlpProtocol::Http`] (`http/json`) and [`OtlpProtocol::HttpProtobuf`]
//! (`http/protobuf`) build an `OtlpHttpExporter` with the matching encoding —
//! JSON is the default, protobuf is opt-in. [`OtlpProtocol::Grpc`] — the
//! configuration default — answers `Error::Unsupported`, because gRPC framing
//! needs `opentelemetry-otlp`, `prost` and `tonic`, none of which the offline
//! build can reach. A configuration value or an
//! `OTEL_EXPORTER_OTLP_PROTOCOL=http/json|http/protobuf` variable selects a
//! supported transport; nothing is downgraded silently. The endpoint, the
//! headers, and the request timeout come from the configuration first, then the
//! `OPENJIUWEN_OTEL_*` variables, then the `OTEL_*` ones, then the built-in
//! defaults described in [`super::env`].
//!
//! [`ExporterType`]: super::config::ExporterType
//! [`ExporterType::Memory`]: super::config::ExporterType::Memory
//! [`ExporterType::Console`]: super::config::ExporterType::Console
//! [`ExporterType::None`]: super::config::ExporterType::None
//! [`ExporterType::Otlp`]: super::config::ExporterType::Otlp
//! [`ExporterType::File`]: super::config::ExporterType::File
//! [`OtlpProtocol::Http`]: super::config::OtlpProtocol::Http
//! [`OtlpProtocol::HttpProtobuf`]: super::config::OtlpProtocol::HttpProtobuf
//! [`OtlpProtocol::Grpc`]: super::config::OtlpProtocol::Grpc
//! [`init_otel_tracer`]: self::init_otel_tracer

use std::time::Duration;

#[cfg(feature = "otel")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "otel")]
use std::sync::Arc;

#[cfg(not(feature = "otel"))]
use openjiuwen_common::error::Error;
use openjiuwen_common::error::Result;

#[cfg(feature = "otel")]
use opentelemetry::trace::TracerProvider as _;
#[cfg(feature = "otel")]
use opentelemetry_sdk::trace::{SdkTracer, SdkTracerProvider, SpanExporter};

use super::config::ExporterType;
use super::config::OtelTracerConfig;
#[cfg(feature = "otel")]
use super::config::OtlpProtocol;
#[cfg(feature = "otel")]
use super::otlp::{OtlpEncoding, OtlpHttpExporter};
#[cfg(feature = "otel")]
use super::provider::{
    build_provider, build_provider_without_exporter, CaptureExporter, ConsoleExporter,
};
#[cfg(feature = "otel")]
use crate::observability::file_exporter::TraceFileExporter;

/// Reason reported when the configuration asks for OTLP/gRPC.
#[cfg(feature = "otel")]
const OTLP_GRPC_UNSUPPORTED: &str = "OTLP/gRPC span export is unsupported: gRPC framing needs \
     opentelemetry-otlp, prost, and tonic, which are unavailable offline; use OTLP/HTTP instead, \
     with JSON payloads (OtlpProtocol::Http or OTEL_EXPORTER_OTLP_PROTOCOL=http/json) or \
     protobuf payloads (OtlpProtocol::HttpProtobuf or \
     OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf)";

/// `tracing` target for handle-level failures.
#[cfg(feature = "otel")]
const LOG_TARGET: &str = "openjiuwen_extensions::otel::handle";

/// Why an exporter that only the SDK can implement is refused in a build
/// without the `otel` feature.
///
/// Shared by [`ExporterType::Otlp`] and [`ExporterType::File`], whose request
/// (a collector endpoint, a traces directory) cannot be served at all here, so
/// the caller gets a typed failure instead of an inactive handle.
#[cfg(not(feature = "otel"))]
const OTEL_FEATURE_OFF_REASON: &str = "this build has the `otel` cargo feature off, so no \
     OpenTelemetry provider or exporter exists; enable the `otel` feature or select \
     ExporterType::Memory, ExporterType::Console or ExporterType::None";

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
    /// Trace-file exporter, present only for [`ExporterType::File`], so a host
    /// can observe whether span export is actually reaching disk.
    #[cfg(feature = "otel")]
    file: Option<TraceFileExporter>,
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
    /// Flushes pending spans and then shuts the provider down, both bounded by
    /// [`OtelTracerConfig::export_timeout_ms`] (minimum 1 ms). A no-op for an
    /// inactive handle.
    pub fn shutdown(&self) {
        self.shutdown_with_timeout(Duration::from_millis(self.config.export_timeout_ms.max(1)));
    }

    /// Flushes pending spans and shuts the private provider down within
    /// `timeout`.
    ///
    /// The flush runs first so spans still sitting in the batch processor are
    /// handed to the exporter (an OTLP batch would otherwise be dropped), then
    /// the provider stops its processors and waits at most `timeout` for them.
    /// Every failure is logged at `warn` on target
    /// `openjiuwen_extensions::otel::handle` and never panics.
    ///
    /// Idempotent: only the first call does any work, and further calls return
    /// immediately. A no-op for an inactive handle, including the handle the
    /// no-feature seam returns.
    pub fn shutdown_with_timeout(&self, timeout: Duration) {
        #[cfg(feature = "otel")]
        {
            if self.is_shutdown() || self.shutdown.swap(true, Ordering::AcqRel) {
                return;
            }
            let Some(provider) = self.provider.as_ref() else {
                return;
            };
            if let Err(error) = provider.force_flush() {
                tracing::warn!(
                    target: LOG_TARGET,
                    %error,
                    "failed to flush the otel tracer provider"
                );
            }
            if let Err(error) = provider.shutdown_with_timeout(timeout) {
                tracing::warn!(
                    target: LOG_TARGET,
                    %error,
                    "failed to shut the otel tracer provider down"
                );
            }
        }
        #[cfg(not(feature = "otel"))]
        {
            // The no-feature seam owns no provider, so there is nothing to flush
            // or stop; keep the signature identical in both builds.
            let _ = (self, timeout);
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

    /// Returns the trace-file exporter, present only when the configuration
    /// asked for [`ExporterType::File`].
    ///
    /// The handle shares the exporter's state with the provider, so the
    /// counters it exposes — `dropped_spans`, `failed_exports`, `last_error` —
    /// report what the live exporter is doing. [`ExporterType::File`] export is
    /// deliberately fail-open (telemetry must never break the caller), so these
    /// accessors are how a host learns that spans are not reaching disk.
    ///
    /// [`ExporterType::File`]: super::config::ExporterType::File
    #[cfg(feature = "otel")]
    #[must_use]
    pub fn file_exporter(&self) -> Option<&TraceFileExporter> {
        self.file.as_ref()
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
            file: None,
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
        file: Option<TraceFileExporter>,
    ) -> Self {
        Self {
            config,
            provider: Some(provider),
            tracer: Some(tracer),
            capture,
            file,
            shutdown: AtomicBool::new(false),
        }
    }
}

/// Initializes the tracer, without the `otel` feature.
///
/// The configuration is validated first, so an invalid setting (for example a
/// `sample_rate` outside `0.0..=1.0`) answers [`Error::Validation`] for every
/// exporter type. The exporter type then decides:
///
/// - [`ExporterType::Memory`], [`ExporterType::Console`] and
///   [`ExporterType::None`] return an inactive handle that keeps the
///   configuration; [`TracerHandle::is_active`] is `false`, which is the
///   documented no-op seam — without the SDK there is no in-process capture and
///   no console sink to attach;
/// - [`ExporterType::Otlp`] and [`ExporterType::File`] answer
///   [`Error::Unsupported`], naming the missing `otel` cargo feature and the
///   exporter types that do work here. Neither a collector endpoint nor a
///   traces directory is ever accepted and silently ignored.
///
/// # Errors
///
/// Returns [`openjiuwen_common::error::Error::validation`] when the
/// configuration is invalid (checked before the exporter type, so it wins in
/// both builds), and
/// [`openjiuwen_common::error::Error::Unsupported`] for
/// [`ExporterType::Otlp`] and [`ExporterType::File`].
#[cfg(not(feature = "otel"))]
pub fn init_otel_tracer(config: OtelTracerConfig) -> Result<TracerHandle> {
    config.validate()?;
    match config.exporter_type {
        ExporterType::Otlp => Err(Error::unsupported(format!(
            "the OTLP/HTTP span exporter is unavailable: {OTEL_FEATURE_OFF_REASON}"
        ))),
        ExporterType::File => Err(Error::unsupported(format!(
            "the JSONL trace-file exporter is unavailable: {OTEL_FEATURE_OFF_REASON}"
        ))),
        ExporterType::Memory | ExporterType::Console | ExporterType::None => {
            Ok(TracerHandle::inactive(config))
        }
    }
}

/// Initializes the tracer and returns its private provider handle.
///
/// `Memory` attaches a [`CaptureExporter`], `Console` exports one `tracing`
/// line per span at `debug`, `File` appends one OTLP JSON line per span to
/// `<traces_dir>/traces-<YYYY-MM-DD>.jsonl` without a collector (see
/// `crate::observability::file_exporter`), `None` builds a provider whose spans
/// are dropped, and `Otlp` builds an [`OtlpHttpExporter`] against the resolved
/// collector endpoint, encoding its bodies as JSON ([`OtlpProtocol::Http`]) or
/// protobuf ([`OtlpProtocol::HttpProtobuf`]); see [`super::env`] for the
/// precedence.
///
/// # Errors
///
/// Returns [`openjiuwen_common::error::Error::validation`] when the
/// configuration is invalid, and
/// [`openjiuwen_common::error::Error::Unsupported`] when `Otlp` resolves to
/// [`OtlpProtocol::Grpc`] — gRPC framing needs `tonic`/`prost`, which are
/// unavailable offline; both OTLP/HTTP encodings ([`OtlpProtocol::Http`] and
/// [`OtlpProtocol::HttpProtobuf`]) are supported transports. A rejected
/// endpoint or header surfaces as
/// [`openjiuwen_common::error::Error::Validation`].
#[cfg(feature = "otel")]
pub fn init_otel_tracer(config: OtelTracerConfig) -> Result<TracerHandle> {
    config.validate()?;
    match config.exporter_type {
        ExporterType::Otlp => {
            let exporter = otlp_exporter(&config)?;
            let provider = build_provider(&config, exporter)?;
            let tracer = provider.tracer(config.tracer_name.clone());
            Ok(TracerHandle::active(config, provider, tracer, None, None))
        }
        ExporterType::Memory => {
            let capture = Arc::new(CaptureExporter::new());
            let provider = build_provider(&config, capture.as_ref().clone())?;
            let tracer = provider.tracer(config.tracer_name.clone());
            Ok(TracerHandle::active(
                config,
                provider,
                tracer,
                Some(capture),
                None,
            ))
        }
        ExporterType::Console => {
            let provider = build_provider(&config, ConsoleExporter)?;
            let tracer = provider.tracer(config.tracer_name.clone());
            Ok(TracerHandle::active(config, provider, tracer, None, None))
        }
        ExporterType::File => {
            // One exporter, two handles: the provider drives it and the handle
            // keeps it so a host can read its failure counters.
            let exporter = TraceFileExporter::from_config(&config);
            let provider = build_provider(&config, exporter.clone())?;
            let tracer = provider.tracer(config.tracer_name.clone());
            Ok(TracerHandle::active(
                config,
                provider,
                tracer,
                None,
                Some(exporter),
            ))
        }
        ExporterType::None => {
            let provider = build_provider_without_exporter(&config)?;
            let tracer = provider.tracer(config.tracer_name.clone());
            Ok(TracerHandle::active(config, provider, tracer, None, None))
        }
    }
}

/// Initializes the tracer around a caller-supplied [`SpanExporter`].
///
/// Useful for hosts that already own an exporter (a vendor bridge, a test sink)
/// and for tests that need to observe spans directly. The provider is still
/// private to the returned handle and [`TracerHandle::capture`] stays `None`,
/// because the handle cannot see inside a foreign exporter. The injected
/// exporter wins for every [`ExporterType`], including `Otlp`: the
/// configuration's protocol and endpoint are not consulted, so a `Grpc`
/// configuration still builds a provider around the caller's exporter.
///
/// # Errors
///
/// Returns [`openjiuwen_common::error::Error::validation`] when the
/// configuration is invalid.
#[cfg(feature = "otel")]
pub fn init_otel_tracer_with_exporter<E>(
    config: OtelTracerConfig,
    exporter: E,
) -> Result<TracerHandle>
where
    E: SpanExporter + 'static,
{
    config.validate()?;
    let provider = build_provider(&config, exporter)?;
    let tracer = provider.tracer(config.tracer_name.clone());
    Ok(TracerHandle::active(config, provider, tracer, None, None))
}

/// Builds the OTLP/HTTP exporter described by `config`.
///
/// The transport is resolved first: an explicit [`OtlpProtocol::Http`] or
/// [`OtlpProtocol::HttpProtobuf`] wins, and the matching environment spellings
/// select them too, while a configuration that still asks for
/// [`OtlpProtocol::Grpc`] is rejected instead of being downgraded silently. The
/// endpoint, headers, and request timeout follow the precedence documented in
/// [`super::env`]; `export_timeout_ms` bounds one HTTP request.
#[cfg(feature = "otel")]
fn otlp_exporter(config: &OtelTracerConfig) -> Result<OtlpHttpExporter> {
    let protocol = super::env::resolve_protocol(config.protocol)?;
    let endpoint = super::env::resolve_endpoint(config.exporter_endpoint.as_deref());
    let headers = super::env::resolve_headers(&config.headers);
    let timeout = Duration::from_millis(config.export_timeout_ms.max(1));
    match protocol {
        OtlpProtocol::Http => OtlpHttpExporter::new(endpoint, headers, timeout),
        OtlpProtocol::HttpProtobuf => {
            OtlpHttpExporter::with_encoding(endpoint, headers, timeout, OtlpEncoding::Protobuf)
        }
        OtlpProtocol::Grpc => Err(openjiuwen_common::error::Error::unsupported(
            OTLP_GRPC_UNSUPPORTED,
        )),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::otel::config::ExporterType;
    #[cfg(feature = "otel")]
    use crate::otel::config::OtlpProtocol;
    #[cfg(not(feature = "otel"))]
    use openjiuwen_common::error::ErrorKind;

    // port of tests/unit_tests/extensions/tracer_otel/test_setup.py

    /// Configuration that selects a capturing provider.
    fn memory_config() -> OtelTracerConfig {
        OtelTracerConfig::default().with_exporter_type(ExporterType::Memory)
    }

    /// Configuration that selects the supported OTLP transport.
    ///
    /// `127.0.0.1:9` is a valid, local, and reliably closed endpoint: building
    /// the exporter never connects, and no test creates a span, so the suite
    /// stays offline.
    #[cfg(feature = "otel")]
    fn otlp_http_config() -> OtelTracerConfig {
        OtelTracerConfig {
            protocol: OtlpProtocol::Http,
            exporter_endpoint: Some("http://127.0.0.1:9".to_owned()),
            export_timeout_ms: 500,
            ..OtelTracerConfig::default()
        }
        .with_exporter_type(ExporterType::Otlp)
    }

    /// The same offline config, selecting the protobuf encoding.
    #[cfg(feature = "otel")]
    fn otlp_http_protobuf_config() -> OtelTracerConfig {
        OtelTracerConfig {
            protocol: OtlpProtocol::HttpProtobuf,
            ..otlp_http_config()
        }
    }

    #[test]
    fn memory_exporter_matches_the_feature_configuration() {
        let handle = init_otel_tracer(memory_config()).unwrap();
        assert_eq!(handle.is_active(), cfg!(feature = "otel"));
        assert_eq!(handle.tracer_name(), "openjiuwen.tracer.otel");
        assert_eq!(handle.config().exporter_type, ExporterType::Memory);
    }

    #[test]
    fn memory_console_and_none_initialize_in_both_builds() {
        for exporter_type in [
            ExporterType::None,
            ExporterType::Memory,
            ExporterType::Console,
        ] {
            let config = OtelTracerConfig::default().with_exporter_type(exporter_type);
            let handle = init_otel_tracer(config).unwrap();
            assert_eq!(handle.is_active(), cfg!(feature = "otel"));
            assert_eq!(handle.config().exporter_type, exporter_type);
        }
    }

    #[cfg(feature = "otel")]
    #[test]
    fn file_exporter_initializes_with_the_feature() {
        // The feature build owns a real provider and never needs a collector.
        // The directory stays inside a temporary dir so the test never writes
        // into the workspace.
        let dir = tempfile::tempdir().unwrap();
        let config = OtelTracerConfig {
            traces_dir: dir.path().to_string_lossy().into_owned(),
            file_retention_days: 0,
            ..OtelTracerConfig::default()
        }
        .with_exporter_type(ExporterType::File);
        let handle = init_otel_tracer(config).unwrap();
        assert!(handle.is_active());
        assert_eq!(handle.config().exporter_type, ExporterType::File);
        // The handle carries the exporter so a host can observe span loss.
        let exporter = handle
            .file_exporter()
            .expect("a File handle exposes its exporter");
        assert_eq!(exporter.root_dir(), dir.path());
        assert_eq!(exporter.dropped_spans(), 0);
        assert_eq!(exporter.failed_exports(), 0);
        assert_eq!(exporter.last_error(), None);
        handle.force_flush();
        handle.shutdown();
        assert!(!handle.is_active());
    }

    #[cfg(feature = "otel")]
    #[test]
    fn non_file_handles_expose_no_file_exporter() {
        for exporter_type in [
            ExporterType::Memory,
            ExporterType::Console,
            ExporterType::None,
        ] {
            let config = OtelTracerConfig::default().with_exporter_type(exporter_type);
            let handle = init_otel_tracer(config).unwrap();
            assert!(handle.file_exporter().is_none(), "{exporter_type:?}");
        }
    }

    #[cfg(not(feature = "otel"))]
    #[test]
    fn otlp_and_file_are_unsupported_without_the_feature() {
        // A collector endpoint and a traces directory are genuine requests to
        // export telemetry, and this build has neither a provider nor an
        // exporter, so both fail with the missing cargo feature instead of
        // handing back an inactive handle that pretends export is on.
        let dir = tempfile::tempdir().unwrap();
        let file_config = OtelTracerConfig {
            traces_dir: dir.path().to_string_lossy().into_owned(),
            file_retention_days: 0,
            ..OtelTracerConfig::default()
        }
        .with_exporter_type(ExporterType::File);
        let otlp_config = OtelTracerConfig {
            exporter_endpoint: Some("http://127.0.0.1:9".to_owned()),
            ..OtelTracerConfig::default()
        }
        .with_exporter_type(ExporterType::Otlp);

        for (exporter_type, config) in [
            (ExporterType::Otlp, otlp_config),
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
            dir.path().read_dir().unwrap().next().is_none(),
            "a rejected File configuration must not create or write trace files"
        );
    }

    #[cfg(not(feature = "otel"))]
    #[test]
    fn invalid_configuration_wins_over_the_missing_feature() {
        // Validation runs before the exporter dispatch, so `Otlp` still reports
        // the configuration error rather than `Unsupported`.
        let config = OtelTracerConfig {
            sample_rate: 1.5,
            ..OtelTracerConfig::default()
        }
        .with_exporter_type(ExporterType::Otlp);
        let error = init_otel_tracer(config).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("sample_rate must be between"));
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
        handle.shutdown_with_timeout(Duration::from_millis(50));
        assert!(!handle.is_active());
        // Post-shutdown calls stay harmless.
        handle.force_flush();
        handle.shutdown();
    }

    #[cfg(feature = "otel")]
    #[test]
    fn otlp_http_exporter_builds_an_active_handle_without_capture() {
        // A guard keeps every OTLP variable clean, so the resolved endpoint and
        // protocol are the ones asserted here. No span is created, so no HTTP
        // request is ever made.
        let _guard = crate::otel::env::test_support::EnvGuard::new();
        let handle = init_otel_tracer(otlp_http_config()).unwrap();
        assert!(handle.is_active());
        assert!(handle.provider().is_some());
        assert!(handle.tracer().is_some());
        assert!(
            handle.capture().is_none(),
            "the OTLP exporter exposes no capture API"
        );
        assert_eq!(handle.config().exporter_type, ExporterType::Otlp);

        handle.shutdown_with_timeout(Duration::from_millis(200));
        assert!(!handle.is_active(), "shutdown deactivates the handle");
    }

    #[cfg(feature = "otel")]
    #[test]
    fn otlp_http_protobuf_exporter_encodes_bodies_as_protobuf() {
        // Building the exporter starts its worker thread but performs no I/O,
        // so no collector has to listen on the endpoint.
        let _guard = crate::otel::env::test_support::EnvGuard::new();
        let config = otlp_http_protobuf_config();

        let exporter = otlp_exporter(&config).unwrap();
        assert_eq!(exporter.content_type(), "application/x-protobuf");
        let debug = format!("{exporter:?}");
        assert!(
            debug.contains("encoding: Protobuf"),
            "unexpected exporter: {debug}"
        );
        drop(exporter);

        let handle = init_otel_tracer(config).unwrap();
        assert!(handle.is_active());
        assert!(handle.provider().is_some());
        assert!(handle.capture().is_none());
        handle.shutdown_with_timeout(Duration::from_millis(200));
        assert!(!handle.is_active());
    }

    #[cfg(feature = "otel")]
    #[test]
    fn otlp_grpc_is_rejected_with_the_http_alternative() {
        let _guard = crate::otel::env::test_support::EnvGuard::new();
        let config = OtelTracerConfig::default().with_exporter_type(ExporterType::Otlp);
        assert_eq!(
            config.protocol,
            OtlpProtocol::Grpc,
            "the configuration default is the unsupported transport"
        );

        let error = init_otel_tracer(config).unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Unsupported
        );
        let message = error.to_string();
        assert!(message.contains("gRPC"), "unexpected message: {message}");
        assert!(
            message.contains("protobuf"),
            "the reason must name the payload encoding: {message}"
        );
        assert!(
            message.contains("http/json"),
            "the reason must name the JSON alternative: {message}"
        );
        assert!(
            message.contains("http/protobuf"),
            "the reason must name the protobuf alternative: {message}"
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
    fn injected_exporter_is_accepted_for_otlp() {
        use crate::otel::provider::CaptureExporter;
        use opentelemetry::trace::{Span as _, Tracer as _};

        // An injected exporter owns the export path, so the `Grpc` protocol in
        // the default configuration is never consulted.
        let capture = CaptureExporter::new();
        let config = OtelTracerConfig::default().with_exporter_type(ExporterType::Otlp);
        let handle = init_otel_tracer_with_exporter(config, capture.clone()).unwrap();
        assert!(handle.is_active());
        assert!(handle.capture().is_none());

        handle.tracer().unwrap().start("otlp.injected").end();
        handle.force_flush();
        assert_eq!(capture.names(), vec!["otlp.injected".to_owned()]);
    }

    #[cfg(feature = "otel")]
    #[test]
    fn shutdown_with_timeout_is_idempotent() {
        use opentelemetry::trace::{Span as _, Tracer as _};

        let handle = init_otel_tracer(memory_config()).unwrap();
        handle.tracer().unwrap().start("before.shutdown").end();

        handle.shutdown_with_timeout(Duration::from_millis(250));
        assert!(!handle.is_active());
        // A second call, of either shutdown flavour, is harmless.
        handle.shutdown_with_timeout(Duration::from_millis(250));
        handle.shutdown();
        assert!(!handle.is_active());
    }
}
