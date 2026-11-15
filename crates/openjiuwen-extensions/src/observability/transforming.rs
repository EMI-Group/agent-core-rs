//! Exporter-side span transformation adapters.
//!
//! Port of `openjiuwen/extensions/observability/exporters/transforming.py`
//! (v0.1.20).
//!
//! The collection layer records only backend-neutral canonical attributes.
//! Adapters built on [`TransformingSpanExporter`] derive backend-specific
//! projections (e.g. `langfuse.*`) on the export path, after the span has
//! ended, so the derived fields never consume collection-time `SpanLimits`
//! quota — `max_attributes`, `max_attribute_length`, `max_events` — and never
//! mutate the original span.
//! Only the wrapped exporter is affected: nothing else in the processor chain
//! sees the projected spans.
//!
//! # Failure policy
//!
//! A transform failure is never allowed to drop telemetry: the failure is
//! logged as a warning and the original, untransformed span is exported
//! instead.
//! [`SpanTransform::transform`] is therefore fallible
//! ([`openjiuwen_common::error::Result`]) rather than panicking, and the
//! exporter handles the error one span at a time.
use openjiuwen_common::error::Result;
use opentelemetry::KeyValue;
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::trace::{SpanData, SpanExporter};
use opentelemetry_sdk::Resource;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, warn};
/// `tracing` target of the exporter and its failure warnings.
const TARGET: &str = "openjiuwen_extensions::observability::transforming";
/// A read-only span transform: returns a new span, never mutates the input.
///
/// Python's `SpanTransform = Callable[[ReadableSpan], ReadableSpan]`.
/// The blanket implementation below accepts a closure, which is the ordinary
/// way to build one:
///
/// ```
/// # #[cfg(feature = "otel")]
/// # {
/// use openjiuwen_common::error::Result;
/// use openjiuwen_extensions::observability::transforming::SpanTransform;
/// use opentelemetry_sdk::trace::SpanData;
///
/// // A closure is a `SpanTransform` through the blanket implementation below.
/// let identity = |span: SpanData| -> Result<SpanData> { Ok(span) };
/// let transform: &dyn SpanTransform = &identity;
/// # let _ = transform;
/// # }
/// ```
pub trait SpanTransform: Send + Sync {
    /// Returns the span to export in place of `span`.
    ///
    /// # Errors
    ///
    /// Returns the projection's own failure when the span cannot be projected;
    /// the exporter then logs a warning and exports `span` unchanged, so
    /// telemetry is never dropped because a transform failed.
    fn transform(&self, span: SpanData) -> Result<SpanData>;
}
impl<F> SpanTransform for F
where
    F: Fn(SpanData) -> Result<SpanData> + Send + Sync,
{
    fn transform(&self, span: SpanData) -> Result<SpanData> {
        self(span)
    }
}
/// Returns a full copy of `span` with `attributes` replacing its own.
///
/// Identity is preserved verbatim: trace/span id, parent span id and its remote
/// flag, trace flags and trace state (both inside the span context), span kind,
/// timestamps, status, events, links, dropped counts and instrumentation scope
/// all reference the original values.
/// The resource is not part of `SpanData` in the SDK this crate targets
/// (Python's `ReadableSpan.resource` has no counterpart); the exporter's own
/// resource is forwarded verbatim by `set_resource`.
/// Only the attribute map is replaced, which is what keeps a projection from
/// touching the span the collection layer already recorded.
#[must_use]
pub fn clone_span_with_attributes(span: &SpanData, attributes: Vec<KeyValue>) -> SpanData {
    SpanData {
        span_context: span.span_context.clone(),
        parent_span_id: span.parent_span_id,
        parent_span_is_remote: span.parent_span_is_remote,
        span_kind: span.span_kind.clone(),
        name: span.name.clone(),
        start_time: span.start_time,
        end_time: span.end_time,
        attributes,
        dropped_attributes_count: span.dropped_attributes_count,
        events: span.events.clone(),
        links: span.links.clone(),
        status: span.status.clone(),
        instrumentation_scope: span.instrumentation_scope.clone(),
    }
}
/// Wraps one exporter behind a per-span read-only transformation.
///
/// Each ended span is copied and handed to the transform; the wrapped exporter
/// only ever sees the projected copies.
/// Python's `TransformingSpanExporter`.
#[must_use]
pub struct TransformingSpanExporter<E> {
    /// The exporter that receives the projected spans.
    inner: E,
    /// The per-span projection.
    transform: Arc<dyn SpanTransform>,
}
impl<E> TransformingSpanExporter<E> {
    /// Wraps `inner` behind `transform`.
    pub fn new(inner: E, transform: impl SpanTransform + 'static) -> Self {
        Self {
            inner,
            transform: Arc::new(transform),
        }
    }
    /// Returns the wrapped exporter.
    pub fn inner(&self) -> &E {
        &self.inner
    }
    /// Returns the transformation applied to every exported span.
    pub fn transform(&self) -> &Arc<dyn SpanTransform> {
        &self.transform
    }
    /// Returns the span to hand to the wrapped exporter.
    ///
    /// A failing transform logs one warning and yields the original span, so a
    /// projection bug costs a projection and never the telemetry.
    /// The span name is logged at `debug` only, so a failing transform cannot
    /// fill the warning log with high-cardinality span names.
    fn project(&self, span: SpanData) -> SpanData {
        match self.transform.transform(span.clone()) {
            Ok(projected) => projected,
            Err(error) => {
                warn!(
                    target: TARGET,
                    exporter = std::any::type_name::<E>(),
                    error = %error,
                    "span transform failed - exporting the original span",
                );
                debug!(target: TARGET, span = %span.name, "the span whose transform failed");
                span
            }
        }
    }
}
impl<E: fmt::Debug> fmt::Debug for TransformingSpanExporter<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TransformingSpanExporter")
            .field("inner", &self.inner)
            .field("transform", &"<dyn SpanTransform>")
            .finish()
    }
}
impl<E: SpanExporter> SpanExporter for TransformingSpanExporter<E> {
    async fn export(&self, batch: Vec<SpanData>) -> OTelSdkResult {
        let projected = batch
            .into_iter()
            .map(|span| self.project(span))
            .collect::<Vec<SpanData>>();
        self.inner.export(projected).await
    }
    fn force_flush(&mut self) -> OTelSdkResult {
        self.inner.force_flush()
    }
    fn shutdown_with_timeout(&mut self, timeout: Duration) -> OTelSdkResult {
        self.inner.shutdown_with_timeout(timeout)
    }
    fn set_resource(&mut self, resource: &Resource) {
        self.inner.set_resource(resource);
    }
}
#[cfg(test)]
mod tests {
    //! Unit tests of the transforming exporter.
    //!
    //! The export path is driven with `futures::executor::block_on` from plain
    //! `#[test]` functions, the way the OTLP exporter suite does it: the SDK's
    //! batch processor runs `export()` on a thread without a tokio reactor, so
    //! the wrapper must not need one either.
    //!
    //! A test module owns its panics: a failed `unwrap`/`expect` is an assertion.
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::{clone_span_with_attributes, TransformingSpanExporter};
    use crate::otel::provider::CaptureExporter;
    use futures::executor::block_on;
    use opentelemetry::trace::{
        Event, Link, SpanContext, SpanId, SpanKind, Status, TraceFlags, TraceId, TraceState,
    };
    use opentelemetry::{InstrumentationScope, KeyValue};
    use opentelemetry_sdk::error::{OTelSdkError, OTelSdkResult};
    use opentelemetry_sdk::trace::{SpanData, SpanEvents, SpanExporter, SpanLinks};
    use opentelemetry_sdk::Resource;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};
    /// The span every test projects: every field carries a non-default value so
    /// the copy is proven to preserve each one.
    fn sample_span() -> SpanData {
        SpanData {
            span_context: SpanContext::new(
                TraceId::from_bytes([0x11; 16]),
                SpanId::from_bytes([0x22; 8]),
                TraceFlags::SAMPLED,
                false,
                TraceState::from_key_value([("vendor", "state")]).unwrap(),
            ),
            parent_span_id: SpanId::from_bytes([0x33; 8]),
            parent_span_is_remote: true,
            span_kind: SpanKind::Client,
            name: "chat model".into(),
            start_time: UNIX_EPOCH + Duration::from_secs(1_700_000_000),
            end_time: UNIX_EPOCH + Duration::from_secs(1_700_000_001),
            attributes: vec![KeyValue::new("gen_ai.operation.name", "chat")],
            dropped_attributes_count: 2,
            events: span_events(),
            links: span_links(),
            status: Status::error("boom"),
            instrumentation_scope: InstrumentationScope::builder("openjiuwen.test").build(),
        }
    }
    /// One event and a non-zero dropped count, so event preservation is visible.
    fn span_events() -> SpanEvents {
        let mut events = SpanEvents::default();
        events.events = vec![Event::new(
            "exception",
            SystemTime::UNIX_EPOCH,
            vec![KeyValue::new("exception.type", "TimeoutError")],
            0,
        )];
        events.dropped_count = 1;
        events
    }
    /// One link and a non-zero dropped count.
    fn span_links() -> SpanLinks {
        let mut links = SpanLinks::default();
        links.links = vec![Link::new(
            SpanContext::new(
                TraceId::from_bytes([0x44; 16]),
                SpanId::from_bytes([0x55; 8]),
                TraceFlags::SAMPLED,
                true,
                TraceState::NONE,
            ),
            Vec::new(),
            0,
        )];
        links.dropped_count = 3;
        links
    }
    /// The span captured by `capture`, which must hold exactly one span.
    fn only_span(capture: &CaptureExporter) -> SpanData {
        let mut spans = capture.spans();
        assert_eq!(spans.len(), 1, "expected exactly one exported span");
        spans.remove(0)
    }
    /// Exports one batch through the wrapper; a failed transform is not a failed export.
    fn export_with(exporter: &TransformingSpanExporter<CaptureExporter>, batch: Vec<SpanData>) {
        let result = block_on(SpanExporter::export(exporter, batch));
        assert!(
            result.is_ok(),
            "a transform failure must not fail the export"
        );
    }
    /// An exporter that only counts the calls delegated to it.
    #[derive(Debug, Clone, Default)]
    struct CountingExporter {
        /// Number of `export` calls.
        exports: Arc<AtomicUsize>,
        /// Number of `force_flush` calls.
        flushes: Arc<AtomicUsize>,
        /// Number of `shutdown_with_timeout` calls.
        shutdowns: Arc<AtomicUsize>,
        /// Resource last handed to `set_resource`.
        resource: Arc<Mutex<Option<Resource>>>,
        /// Whether `export` reports a failure.
        fail: bool,
    }
    impl SpanExporter for CountingExporter {
        fn export(
            &self,
            _batch: Vec<SpanData>,
        ) -> impl std::future::Future<Output = OTelSdkResult> + Send {
            self.exports.fetch_add(1, Ordering::SeqCst);
            if self.fail {
                return std::future::ready(Err(OTelSdkError::InternalFailure(
                    "inner failure".to_owned(),
                )));
            }
            std::future::ready(Ok(()))
        }
        fn force_flush(&mut self) -> OTelSdkResult {
            self.flushes.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn shutdown_with_timeout(&mut self, _timeout: Duration) -> OTelSdkResult {
            self.shutdowns.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn set_resource(&mut self, resource: &Resource) {
            *self.resource.lock().expect("the test lock is not poisoned") = Some(resource.clone());
        }
    }
    #[test]
    fn cloning_replaces_only_the_attributes() {
        let span = sample_span();
        let projected = clone_span_with_attributes(
            &span,
            vec![
                KeyValue::new("gen_ai.operation.name", "chat"),
                KeyValue::new("langfuse.observation.type", "generation"),
            ],
        );
        assert_eq!(projected.span_context, span.span_context);
        assert_eq!(projected.parent_span_id, span.parent_span_id);
        assert_eq!(projected.parent_span_is_remote, span.parent_span_is_remote);
        assert_eq!(projected.span_kind, span.span_kind);
        assert_eq!(projected.name, span.name);
        assert_eq!(projected.start_time, span.start_time);
        assert_eq!(projected.end_time, span.end_time);
        assert_eq!(projected.status, span.status);
        assert_eq!(projected.events, span.events);
        assert_eq!(projected.links, span.links);
        assert_eq!(
            projected.dropped_attributes_count,
            span.dropped_attributes_count
        );
        assert_eq!(projected.instrumentation_scope, span.instrumentation_scope);
        assert_eq!(projected.attributes.len(), 2);
        // The input span is untouched: the projection never mutates it.
        assert_eq!(span.attributes.len(), 1);
        assert_eq!(span.attributes[0].value, opentelemetry::Value::from("chat"));
    }
    #[test]
    fn a_projection_adds_attributes_on_the_export_path() {
        let capture = CaptureExporter::new();
        let exporter = TransformingSpanExporter::new(capture.clone(), |span: SpanData| {
            let mut attributes = span.attributes.clone();
            attributes.push(KeyValue::new("langfuse.observation.type", "generation"));
            Ok(clone_span_with_attributes(&span, attributes))
        });
        export_with(&exporter, vec![sample_span()]);
        let projected = only_span(&capture);
        assert_eq!(projected.name, "chat model");
        let keys: Vec<&str> = projected
            .attributes
            .iter()
            .map(|pair| pair.key.as_str())
            .collect();
        assert_eq!(
            keys,
            vec!["gen_ai.operation.name", "langfuse.observation.type"]
        );
        assert_eq!(projected.status, Status::error("boom"));
        assert_eq!(projected.dropped_attributes_count, 2);
    }
    #[test]
    fn a_failing_transform_exports_the_original_span() {
        let capture = CaptureExporter::new();
        let exporter = TransformingSpanExporter::new(capture.clone(), |_span: SpanData| {
            Err(openjiuwen_common::error::Error::validation(
                "projection exploded",
            ))
        });
        let original = sample_span();
        export_with(&exporter, vec![original.clone()]);
        assert_eq!(only_span(&capture), original);
    }
    #[test]
    fn a_failing_transform_only_costs_its_own_span() {
        let capture = CaptureExporter::new();
        let exporter = TransformingSpanExporter::new(capture.clone(), |mut span: SpanData| {
            if span.name == "fails" {
                return Err(openjiuwen_common::error::Error::validation(
                    "projection exploded",
                ));
            }
            span.attributes.push(KeyValue::new("projected", "yes"));
            Ok(span)
        });
        let mut failing = sample_span();
        failing.name = "fails".into();
        let mut healthy = sample_span();
        healthy.name = "works".into();
        export_with(&exporter, vec![failing, healthy]);
        let spans = capture.spans();
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].attributes, sample_span().attributes);
        assert_eq!(spans[1].attributes.len(), 2);
    }
    #[test]
    fn an_empty_batch_is_still_forwarded() {
        let inner = CountingExporter::default();
        let exporter = TransformingSpanExporter::new(inner.clone(), |span: SpanData| Ok(span));
        let result = block_on(SpanExporter::export(&exporter, Vec::new()));
        assert!(result.is_ok());
        assert_eq!(inner.exports.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn the_transform_is_reachable_for_direct_use() {
        // The accessor hands out the very transformation the exporter applies.
        let exporter =
            TransformingSpanExporter::new(CaptureExporter::new(), |mut span: SpanData| {
                span.attributes.push(KeyValue::new("projected", "yes"));
                Ok(span)
            });
        let projected = exporter.transform().transform(sample_span()).unwrap();
        assert_eq!(projected.attributes.len(), 2);
    }
    #[test]
    fn the_wrapper_reports_the_inner_export_result() {
        let inner = CountingExporter {
            fail: true,
            ..CountingExporter::default()
        };
        let exporter = TransformingSpanExporter::new(inner.clone(), |span: SpanData| Ok(span));
        let result = block_on(SpanExporter::export(&exporter, vec![sample_span()]));
        assert!(matches!(result, Err(OTelSdkError::InternalFailure(_))));
        assert_eq!(inner.exports.load(Ordering::SeqCst), 1);
    }
    #[test]
    fn the_delegate_methods_forward_verbatim() {
        let inner = CountingExporter::default();
        let mut exporter = TransformingSpanExporter::new(inner.clone(), |span: SpanData| Ok(span));
        let resource = Resource::builder_empty()
            .with_attributes([KeyValue::new("service.name", "transforming-test")])
            .build();
        exporter.set_resource(&resource);
        assert!(exporter.force_flush().is_ok());
        assert!(exporter
            .shutdown_with_timeout(Duration::from_millis(5))
            .is_ok());
        assert!(exporter.shutdown().is_ok());
        assert_eq!(inner.flushes.load(Ordering::SeqCst), 1);
        // `shutdown` forwards to `shutdown_with_timeout` and adds one more call.
        assert_eq!(inner.shutdowns.load(Ordering::SeqCst), 2);
        let recorded = inner
            .resource
            .lock()
            .expect("the test lock is not poisoned")
            .clone();
        assert_eq!(recorded, Some(resource));
        assert_eq!(exporter.inner().exports.load(Ordering::SeqCst), 0);
    }
    #[test]
    fn the_wrapper_is_debug_printable_and_exposes_its_parts() {
        let inner = CountingExporter::default();
        let exporter = TransformingSpanExporter::new(inner, |span: SpanData| Ok(span));
        let printed = format!("{exporter:?}");
        assert!(printed.contains("TransformingSpanExporter"));
        assert!(printed.contains("<dyn SpanTransform>"));
        assert_eq!(exporter.inner().exports.load(Ordering::SeqCst), 0);
    }
}
