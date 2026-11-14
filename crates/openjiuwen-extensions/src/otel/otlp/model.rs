//! Span grouping shared by the OTLP body encoders.
//!
//! Both encodings (`json.rs` and `protobuf.rs`) describe the same
//! `ExportTraceServiceRequest`: a single `ResourceSpans` entry whose `spans`
//! are split into one `ScopeSpans` entry per instrumentation scope. The
//! grouping lives here so the two encoders cannot drift apart.

use opentelemetry::InstrumentationScope;
use opentelemetry_sdk::trace::SpanData;

/// One instrumentation scope with the spans it produced, in export order.
pub(super) struct ScopeGroup<'a> {
    /// The scope every span in the group belongs to; the reference comes from
    /// the first span that opened the group.
    pub(super) scope: &'a InstrumentationScope,
    /// The group's spans, in the order the exporter received them.
    pub(super) spans: Vec<&'a SpanData>,
}

/// Groups spans by their `(name, version)` instrumentation scope.
///
/// There is one group per distinct pair, groups keep first-seen order, and each
/// group keeps the export order of its spans — exactly the shape OTLP expects
/// inside `ResourceSpans`, and the shape the JSON encoder produced before the
/// encoders shared this module.
pub(super) fn scope_groups(spans: &[SpanData]) -> Vec<ScopeGroup<'_>> {
    let mut groups: Vec<ScopeGroup<'_>> = Vec::new();
    for span in spans {
        let scope = &span.instrumentation_scope;
        if let Some(group) = groups.iter_mut().find(|group| {
            group.scope.name() == scope.name() && group.scope.version() == scope.version()
        }) {
            group.spans.push(span);
        } else {
            groups.push(ScopeGroup {
                scope,
                spans: vec![span],
            });
        }
    }
    groups
}
