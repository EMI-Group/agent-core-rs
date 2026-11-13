//! Provider construction: resource, sampler, span limits, and exporters.
//!
//! Port of `_build_provider` from
//! `openjiuwen/extensions/tracer_otel/setup.py` (v0.1.20). Python returns a
//! bare `Tracer`; Rust keeps the private `SdkTracerProvider` alive in
//! [`super::handle::TracerHandle`], so the tracer, its exporters, and shutdown
//! stay together and nothing is registered on the process-global
//! `opentelemetry::global` provider.
//!
//! Documented deviations from the Python module:
//!
//! - The OTLP exporter is not built here: [`super::handle::init_otel_tracer`]
//!   resolves the transport and endpoint, then hands an `OtlpHttpExporter` to
//!   `build_provider` like any other exporter. Hosts that own an exporter
//!   inject it through [`super::handle::init_otel_tracer_with_exporter`].
//! - The console exporter writes one `tracing` line per span at `debug`
//!   instead of Python's `ConsoleSpanExporter` stdout dump (libraries never
//!   write to stdout).
//! - `export_timeout_ms` sizes one OTLP HTTP request (see
//!   [`super::handle`]), not the batch processor: `BatchConfigBuilder`'s
//!   `with_max_export_timeout` is gated behind the
//!   `experimental_trace_batch_span_processor_with_async_runtime` feature of
//!   `opentelemetry_sdk` 0.31, which this crate does not enable, so the batch
//!   processor keeps the SDK default (or `OTEL_BSP_EXPORT_TIMEOUT`).
//! - `max_attr_length` is not a provider knob: 0.31's `SpanLimits` has no
//!   attribute-value-length field, so the cap is enforced by
//!   [`super::redaction::truncate`] when attributes are recorded.

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use openjiuwen_common::error::Result;
use opentelemetry::KeyValue;
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::trace::{
    BatchConfig, BatchConfigBuilder, BatchSpanProcessor, Sampler, SdkTracer, SdkTracerProvider,
    SpanData, SpanExporter, SpanLimits, TracerProviderBuilder,
};
use opentelemetry_sdk::Resource;

use super::config::OtelTracerConfig;

/// `service.version` fallback when the configuration leaves it unset.
const UNKNOWN_SERVICE_VERSION: &str = "unknown";

/// `service.name` resource attribute key.
const SERVICE_NAME_KEY: &str = "service.name";

/// `service.version` resource attribute key.
const SERVICE_VERSION_KEY: &str = "service.version";

/// `k8s.pod.name` resource attribute key.
const K8S_POD_NAME_KEY: &str = "k8s.pod.name";

/// Hostname variable that carries the pod name inside a Kubernetes pod.
const HOSTNAME_ENV: &str = "HOSTNAME";

/// Service variable that proves the process really runs inside Kubernetes.
const KUBERNETES_SERVICE_HOST_ENV: &str = "KUBERNETES_SERVICE_HOST";

/// `tracing` target used by the console exporter.
const CONSOLE_TARGET: &str = "openjiuwen_extensions::otel::provider";

/// Locks a mutex, recovering the data when a previous holder panicked.
///
/// Exporter callbacks are fail-open: a poisoned lock must not turn span export
/// into a panic on the batch-processor thread.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// In-memory [`SpanExporter`] that keeps exported [`SpanData`] for tests and
/// hosts.
///
/// Cloning is cheap: every clone shares the same span buffer, so the handle's
/// snapshot and the provider's exporter always observe the same spans.
#[derive(Debug, Clone, Default)]
pub struct CaptureExporter {
    /// Spans in export order.
    spans: Arc<Mutex<Vec<SpanData>>>,
    /// Resource reported by the provider at build time.
    resource: Arc<Mutex<Option<Resource>>>,
}

impl CaptureExporter {
    /// Creates an empty exporter.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a snapshot of the captured spans, in export order.
    #[must_use]
    pub fn spans(&self) -> Vec<SpanData> {
        lock(&self.spans).clone()
    }

    /// Returns the number of captured spans.
    #[must_use]
    pub fn len(&self) -> usize {
        lock(&self.spans).len()
    }

    /// Reports whether no span has been exported yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        lock(&self.spans).is_empty()
    }

    /// Drops every captured span (the recorded resource is kept).
    pub fn clear(&self) {
        lock(&self.spans).clear();
    }

    /// Returns the names of the captured spans, in export order.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        lock(&self.spans)
            .iter()
            .map(|span| span.name.to_string())
            .collect()
    }

    /// Returns the first captured span with exactly this name.
    #[must_use]
    pub fn span_named(&self, name: &str) -> Option<SpanData> {
        lock(&self.spans)
            .iter()
            .find(|span| span.name == name)
            .cloned()
    }

    /// Returns the resource the provider handed to this exporter.
    ///
    /// `None` until a provider is built around the exporter; this is how tests
    /// assert `service.name`/`service.version` without touching global state.
    #[must_use]
    pub fn resource(&self) -> Option<Resource> {
        lock(&self.resource).clone()
    }

    /// Appends one exported batch to the buffer.
    fn record(&self, batch: Vec<SpanData>) {
        lock(&self.spans).extend(batch);
    }
}

impl SpanExporter for CaptureExporter {
    fn export(
        &self,
        batch: Vec<SpanData>,
    ) -> impl std::future::Future<Output = OTelSdkResult> + Send {
        self.record(batch);
        std::future::ready(Ok(()))
    }

    fn set_resource(&mut self, resource: &Resource) {
        *lock(&self.resource) = Some(resource.clone());
    }

    fn shutdown_with_timeout(&mut self, _timeout: Duration) -> OTelSdkResult {
        Ok(())
    }
}

/// Console [`SpanExporter`]: one `tracing` line per span at `debug`.
///
/// Python's `ConsoleSpanExporter` prints to stdout; a library may not, so the
/// line goes through the crate's `tracing` target instead.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ConsoleExporter;

impl SpanExporter for ConsoleExporter {
    fn export(
        &self,
        batch: Vec<SpanData>,
    ) -> impl std::future::Future<Output = OTelSdkResult> + Send {
        for span in &batch {
            log_span(span);
        }
        std::future::ready(Ok(()))
    }

    fn shutdown_with_timeout(&mut self, _timeout: Duration) -> OTelSdkResult {
        Ok(())
    }
}

/// Writes one `debug` line describing a finished span.
fn log_span(span: &SpanData) {
    let duration = span
        .end_time
        .duration_since(span.start_time)
        .unwrap_or_default();
    let attributes = span
        .attributes
        .iter()
        .map(|pair| format!("{}={}", pair.key, pair.value.as_str()))
        .collect::<Vec<_>>()
        .join(", ");
    tracing::debug!(
        target: CONSOLE_TARGET,
        span = %span.name,
        trace_id = %format!("{:x}", span.span_context.trace_id()),
        span_id = %format!("{:x}", span.span_context.span_id()),
        parent_span_id = %format!("{:x}", span.parent_span_id),
        kind = ?span.span_kind,
        status = ?span.status,
        duration_ms = %duration.as_millis(),
        attribute_count = span.attributes.len(),
        attributes = %attributes,
        "otel span"
    );
}

/// Builds the `tracing` → `OTel` bridge for hosts that already own a
/// subscriber.
///
/// The caller passes the tracer of a [`super::handle::TracerHandle`], so the
/// resulting layer reports into that handle's private provider.
#[must_use]
pub fn tracing_layer<S>(
    tracer: SdkTracer,
) -> tracing_opentelemetry::OpenTelemetryLayer<S, SdkTracer>
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    tracing_opentelemetry::OpenTelemetryLayer::new(tracer)
}

/// Builds a private provider around a caller-supplied exporter.
///
/// The exporter is driven by a [`BatchSpanProcessor`] configured from the
/// configuration's `schedule_delay_millis` and `max_export_batch_size`.
///
/// # Errors
///
/// Reserved for future provider-level failures; the current SDK build cannot
/// fail, so this only forwards configuration problems detected by the caller.
#[allow(clippy::unnecessary_wraps)] // signature mirrors the handle's `Result` API
pub(crate) fn build_provider<E>(config: &OtelTracerConfig, exporter: E) -> Result<SdkTracerProvider>
where
    E: SpanExporter + 'static,
{
    let processor = BatchSpanProcessor::builder(exporter)
        .with_batch_config(batch_config(config))
        .build();
    Ok(provider_builder(config)
        .with_span_processor(processor)
        .build())
}

/// Builds a private provider with no span processor.
///
/// Spans are still created and sampled, but nothing is exported — the Rust
/// equivalent of Python's "no exporter" path.
///
/// # Errors
///
/// Reserved for future provider-level failures; the current SDK build cannot
/// fail.
#[allow(clippy::unnecessary_wraps)] // signature mirrors the handle's `Result` API
pub(crate) fn build_provider_without_exporter(
    config: &OtelTracerConfig,
) -> Result<SdkTracerProvider> {
    Ok(provider_builder(config).build())
}

/// Creates the provider builder shared by both build paths.
fn provider_builder(config: &OtelTracerConfig) -> TracerProviderBuilder {
    SdkTracerProvider::builder()
        .with_resource(build_resource(config))
        .with_sampler(build_sampler(config))
        .with_span_limits(span_limits(config))
}

/// Builds the resource: `service.name`, `service.version`, and the k8s pod
/// name when the process runs inside a pod.
///
/// `Resource::builder` also folds in the SDK defaults and the
/// `OTEL_RESOURCE_ATTRIBUTES`/`OTEL_SERVICE_NAME` variables, matching Python's
/// `Resource.create`. `HOSTNAME` alone is a container or bare-metal hostname,
/// so the pod name is only reported when `KUBERNETES_SERVICE_HOST` is present
/// as well (Python parity).
fn build_resource(config: &OtelTracerConfig) -> Resource {
    let version = config
        .service_version
        .clone()
        .unwrap_or_else(|| UNKNOWN_SERVICE_VERSION.to_owned());
    let mut builder = Resource::builder().with_attributes([
        KeyValue::new(SERVICE_NAME_KEY, config.service_name.clone()),
        KeyValue::new(SERVICE_VERSION_KEY, version),
    ]);
    if let Some(pod_name) = kubernetes_pod_name() {
        builder = builder.with_attributes([KeyValue::new(K8S_POD_NAME_KEY, pod_name)]);
    }
    builder.build()
}

/// Returns the pod name when both Kubernetes markers are set.
fn kubernetes_pod_name() -> Option<String> {
    std::env::var_os(KUBERNETES_SERVICE_HOST_ENV)?;
    std::env::var(HOSTNAME_ENV)
        .ok()
        .filter(|name| !name.is_empty())
}

/// Builds the parent-based ratio sampler Python configures.
fn build_sampler(config: &OtelTracerConfig) -> Sampler {
    Sampler::ParentBased(Box::new(Sampler::TraceIdRatioBased(config.sample_rate)))
}

/// Builds span limits: the attribute cap applies per span and per event, every
/// other limit keeps the SDK default.
fn span_limits(config: &OtelTracerConfig) -> SpanLimits {
    let max_attributes = u32::try_from(config.max_attributes).unwrap_or(u32::MAX);
    SpanLimits {
        max_attributes_per_span: max_attributes,
        max_attributes_per_event: max_attributes,
        ..SpanLimits::default()
    }
}

/// Builds the batch processor configuration.
///
/// `export_timeout_ms` is deliberately absent: the SDK builder method for the
/// batch export timeout is feature-gated (see the module docs), so the batch
/// processor keeps the SDK default and a host that needs to change it sets
/// `OTEL_BSP_EXPORT_TIMEOUT`. The OTLP HTTP request timeout uses
/// `export_timeout_ms` instead, in [`super::handle`].
fn batch_config(config: &OtelTracerConfig) -> BatchConfig {
    BatchConfigBuilder::default()
        .with_scheduled_delay(Duration::from_millis(config.schedule_delay_millis))
        .with_max_export_batch_size(config.max_export_batch_size.max(1))
        .build()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::otel::config::{ExporterType, OtelTracerConfig};
    use crate::otel::handle::{init_otel_tracer, init_otel_tracer_with_exporter};
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry::trace::{Span as _, SpanKind, Status, Tracer as _};
    use opentelemetry::Key;
    use tracing_subscriber::layer::SubscriberExt as _;

    // port of tests/unit_tests/extensions/tracer_otel/test_setup.py
    // (service name/version, batch-processor config, span round-trip cases)

    fn memory_config() -> OtelTracerConfig {
        OtelTracerConfig::default().with_exporter_type(ExporterType::Memory)
    }

    /// Reads a string attribute off a resource.
    fn resource_attribute(resource: &Resource, key: &'static str) -> Option<String> {
        resource
            .get(&Key::from_static_str(key))
            .map(|value| value.as_str().into_owned())
    }

    #[test]
    fn captured_span_reaches_the_exporter_after_flush() {
        let handle = init_otel_tracer(memory_config()).unwrap();
        let tracer = handle.tracer().expect("memory handle must expose a tracer");

        let mut span = tracer.start("chat SupportAgent");
        span.set_attribute(KeyValue::new("gen_ai.operation.name", "chat"));
        span.set_attribute(KeyValue::new("gen_ai.request.model", "gpt-4o"));
        span.set_status(Status::Ok);
        span.end();

        handle.force_flush();

        let capture = handle.capture().expect("memory handle must capture spans");
        assert_eq!(capture.len(), 1);
        assert_eq!(capture.names(), vec!["chat SupportAgent".to_owned()]);

        let exported = capture
            .span_named("chat SupportAgent")
            .expect("span must be exported");
        assert_eq!(exported.span_kind, SpanKind::Internal);
        assert_eq!(exported.status, Status::Ok);
        let operation = exported
            .attributes
            .iter()
            .find(|pair| pair.key == Key::from_static_str("gen_ai.operation.name"))
            .expect("operation attribute must survive");
        assert_eq!(operation.value.as_str(), "chat");

        capture.clear();
        assert!(capture.is_empty());
        assert!(capture.span_named("chat SupportAgent").is_none());
    }

    #[test]
    fn resource_carries_service_name_and_version() {
        let config = memory_config().with_service("my-service", Some("1.2.3".to_owned()));
        let handle = init_otel_tracer(config).unwrap();
        // The batch processor forwards the resource to the exporter on its own
        // thread; a flush makes that hand-off deterministic.
        handle.force_flush();
        let capture = handle.capture().unwrap();

        let resource = capture.resource().expect("provider must set the resource");
        assert_eq!(
            resource_attribute(&resource, SERVICE_NAME_KEY).as_deref(),
            Some("my-service")
        );
        assert_eq!(
            resource_attribute(&resource, SERVICE_VERSION_KEY).as_deref(),
            Some("1.2.3")
        );
    }

    #[test]
    fn missing_service_version_falls_back_to_unknown() {
        let config = memory_config().with_service("svc", None);
        let handle = init_otel_tracer(config).unwrap();
        handle.force_flush();
        let capture = handle.capture().unwrap();

        let resource = capture.resource().expect("provider must set the resource");
        assert_eq!(
            resource_attribute(&resource, SERVICE_VERSION_KEY).as_deref(),
            Some(UNKNOWN_SERVICE_VERSION)
        );
    }

    #[test]
    fn injected_exporter_receives_spans() {
        let capture = CaptureExporter::new();
        let handle = init_otel_tracer_with_exporter(memory_config(), capture.clone()).unwrap();
        assert!(handle.is_active());
        assert!(handle.capture().is_none());

        let tracer = handle.tracer().unwrap();
        tracer.start("memory.read").end();
        handle.force_flush();

        assert_eq!(capture.names(), vec!["memory.read".to_owned()]);
        assert_eq!(capture.len(), 1);
    }

    #[test]
    fn provider_without_exporter_creates_spans_without_exporting() {
        let config = OtelTracerConfig::default();
        let provider = build_provider_without_exporter(&config).unwrap();
        let tracer = provider.tracer("openjiuwen.tracer.otel");
        tracer.start("dropped").end();
        assert!(provider.force_flush().is_ok());
        assert!(provider.shutdown().is_ok());
    }

    #[test]
    fn handles_export_independently() {
        // No global provider registration: two handles keep separate buffers.
        let first = init_otel_tracer(memory_config()).unwrap();
        let second = init_otel_tracer(memory_config()).unwrap();

        first.tracer().unwrap().start("first.span").end();
        first.force_flush();
        second.force_flush();

        assert_eq!(
            first.capture().unwrap().names(),
            vec!["first.span".to_owned()]
        );
        assert!(second.capture().unwrap().is_empty());
    }

    #[test]
    fn zero_sample_rate_drops_spans() {
        let config = OtelTracerConfig {
            sample_rate: 0.0,
            ..memory_config()
        };
        let handle = init_otel_tracer(config).unwrap();
        handle.tracer().unwrap().start("never.sampled").end();
        handle.force_flush();
        assert!(handle.capture().unwrap().is_empty());
    }

    #[test]
    fn tracing_layer_instantiates_against_a_registry() {
        let capture = CaptureExporter::new();
        let config = memory_config();
        let provider = build_provider(&config, capture.clone()).unwrap();
        let tracer = provider.tracer("openjiuwen.tracer.otel");

        let subscriber = tracing_subscriber::registry()
            .with(tracing_layer::<tracing_subscriber::Registry>(tracer));
        tracing::subscriber::with_default(subscriber, || {
            tracing::info_span!("layer_span").in_scope(|| {});
        });

        assert!(provider.force_flush().is_ok());
        assert!(
            capture.names().iter().any(|name| name == "layer_span"),
            "unexpected spans: {:?}",
            capture.names()
        );
    }
}
