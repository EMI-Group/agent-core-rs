//! Per-invoke span state and the agent/workflow span managers.
//!
//! Port of `openjiuwen/extensions/tracer_otel/span_manager.py` (v0.1.20):
//! [`OtelSpanState`] wraps an open `OTel` span with its invoke id and cached
//! start time, [`OtelAgentSpanManager`] keeps the invoke → state map an agent
//! handler uses to parent child spans, and [`OtelWorkflowSpanManager`] adds the
//! three incremental buffers the workflow handler flushes onto the span when the
//! invocation finishes.
//!
//! Python uses instance-level dicts (not `contextvars`) because the handlers run
//! sequentially inside one async context. The Rust handlers are `Send + Sync`
//! and may serve several invocations from different tasks, so the maps sit
//! behind a [`Mutex`]; every method takes `&self`, no lock is ever held across
//! an `await`, and a poisoned lock is recovered rather than propagated
//! (tracing is fail-open). A closure passed to [`OtelAgentSpanManager::with_state`]
//! runs under the lock, so it must not re-enter the same manager.
//!
//! # Deviations from Python
//!
//! - Python's state carries an `otel_context.attach` token and detaches it on
//!   pop; Rust has no ambient context attachment on the session path, so the
//!   state stores no token and the handlers pass the parent [`Context`] from
//!   [`OtelAgentSpanManager::parent_context`] to the tracer instead.
//! - [`OtelAgentSpanManager::has_open_tool_span`] additionally requires the
//!   cached `gen_ai.operation.name` to be `execute_tool` (Python matches any
//!   open span carrying the tool name), so a chat span that mentions a tool
//!   name can never suppress a real tool span (`bug #2046`).
//! - Python reads the tool identity back from `span.attributes`; the SDK span
//!   does not expose them, so [`OtelSpanState`] caches every attribute it
//!   records through [`OtelSpanState::record_attribute`].

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Mutex, MutexGuard};
use std::time::SystemTime;

use opentelemetry::trace::{Span as _, SpanContext, TraceContextExt as _};
use opentelemetry::{Context, KeyValue};
use opentelemetry_sdk::trace::Span;
use serde_json::Value;

use super::semconv::{
    GEN_AI_OPERATION_EXECUTE_TOOL, GEN_AI_OPERATION_NAME, GEN_AI_TOOL_NAME, OJ_TRACE_ID,
};

/// Locks a manager mutex, recovering the data when a previous holder panicked.
///
/// Tracing is fail-open: a poisoned lock must not turn a trace callback into a
/// panic, so the data is used as-is instead of propagating the poison.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// One open span plus the metadata the handlers need after it started.
///
/// Python counterpart: the `OtelSpanState` dataclass. The span is owned, so
/// dropping the state without ending the span leaks it exactly like Python
/// dropping the object; handlers end the span before popping.
pub struct OtelSpanState {
    /// The open span; every attribute write goes through it.
    pub span: Span,
    /// Invoke id this span was opened for.
    pub invoke_id: String,
    /// Local start time, used for the elapsed-time attribute.
    pub start_time: SystemTime,
    /// Reference to the inputs payload, refreshed by the handlers after
    /// transform callbacks mutated it in place.
    pub inputs: Option<Value>,
    /// Owning workflow id, used for cross-workflow isolation.
    pub workflow_id: Option<String>,
    /// Set for tool spans opened by the tool rail rather than the session
    /// tracer, so [`OtelAgentSpanManager::has_open_tool_span`] can tell them
    /// apart (`bug #2046`).
    pub from_tool_rail: bool,
    /// `span.is_recording()` at construction: non-recording spans skip all
    /// attribute serialization and stream buffering.
    pub recorded: bool,
    /// Copy of every attribute recorded through
    /// [`OtelSpanState::record_attribute`], for dedup and classification.
    pub attributes: BTreeMap<String, String>,
}

impl OtelSpanState {
    /// Wraps an open span in a state with an empty cache.
    ///
    /// `recorded` is the span's `is_recording()` result (or `true` when the
    /// caller knows attribute writes land, as Python does on the agent path);
    /// `start_time` is captured from the local clock, because the session span
    /// carries no instantiated timestamp.
    #[must_use]
    pub fn new(span: Span, invoke_id: impl Into<String>, recorded: bool) -> Self {
        Self {
            span,
            invoke_id: invoke_id.into(),
            start_time: SystemTime::now(),
            inputs: None,
            workflow_id: None,
            from_tool_rail: false,
            recorded,
            attributes: BTreeMap::new(),
        }
    }

    /// Wraps an open span, deriving `recorded` from the span itself.
    #[must_use]
    pub fn from_span(span: Span, invoke_id: impl Into<String>) -> Self {
        let recorded = span.is_recording();
        Self::new(span, invoke_id, recorded)
    }

    /// Writes a string attribute and caches it for dedup and classification.
    ///
    /// Attribute writes on a non-recording span are no-ops, but the cache is
    /// still updated so the handlers can classify the invocation.
    pub fn record_attribute(&mut self, key: &'static str, value: impl Into<String>) {
        let value = value.into();
        self.span.set_attribute(KeyValue::new(key, value.clone()));
        self.attributes.insert(key.to_owned(), value);
    }

    /// Returns a cached attribute recorded through
    /// [`Self::record_attribute`].
    #[must_use]
    pub fn attribute(&self, key: &str) -> Option<&str> {
        self.attributes.get(key).map(String::as_str)
    }

    /// Returns the identity of the wrapped span.
    ///
    /// Python counterpart: `span.get_span_context()`; the SDK returns a
    /// reference, so this clones.
    #[must_use]
    pub fn span_context(&self) -> SpanContext {
        self.span.span_context().clone()
    }
}

impl fmt::Debug for OtelSpanState {
    /// Renders the bookkeeping fields only; span internals stay private.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OtelSpanState")
            .field("invoke_id", &self.invoke_id)
            .field("workflow_id", &self.workflow_id)
            .field("from_tool_rail", &self.from_tool_rail)
            .field("recorded", &self.recorded)
            .field("attributes", &self.attributes)
            .finish_non_exhaustive()
    }
}

/// Invoke id → [`OtelSpanState`] for agent handlers.
///
/// Python counterpart: `OtelAgentSpanManager`. The mapping exists so a child
/// span can find its parent's `OTel` context and continue the trace.
#[derive(Debug, Default)]
pub struct OtelAgentSpanManager {
    /// Open spans, keyed by invoke id.
    spans: Mutex<BTreeMap<String, OtelSpanState>>,
}

impl OtelAgentSpanManager {
    /// Creates an empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers an open span, replacing any state already stored under the
    /// same invoke id.
    pub fn push(&self, invoke_id: impl Into<String>, state: OtelSpanState) {
        lock(&self.spans).insert(invoke_id.into(), state);
    }

    /// Removes and returns the state of an invoke id.
    #[must_use]
    pub fn pop(&self, invoke_id: &str) -> Option<OtelSpanState> {
        lock(&self.spans).remove(invoke_id)
    }

    /// Runs `f` against the open state of an invoke id.
    ///
    /// Returns `None` when the invoke id is unknown, so callers never need a
    /// separate lookup. `f` runs under the manager lock and must not call back
    /// into the same manager.
    pub fn with_state<R>(
        &self,
        invoke_id: &str,
        f: impl FnOnce(&mut OtelSpanState) -> R,
    ) -> Option<R> {
        let mut spans = lock(&self.spans);
        spans.get_mut(invoke_id).map(f)
    }

    /// Reports whether an invoke id still has an open span.
    #[must_use]
    pub fn contains(&self, invoke_id: &str) -> bool {
        lock(&self.spans).contains_key(invoke_id)
    }

    /// Reports whether a tool span opened by the tool rail is still open.
    ///
    /// Rail emission is identified by [`OtelSpanState::from_tool_rail`], not by
    /// attribute presence: `gen_ai.tool.call.id` is only written when the model
    /// supplied an id, while the dedup must also cover id-less calls. Matching
    /// uses the cached `gen_ai.tool.name`, scoped by `openjiuwen.trace.id` when
    /// `trace_id` is non-empty so concurrent agents in different traces never
    /// match each other. Used to drop the session tracer's duplicate tool event
    /// for one physical call (`bug #2046`).
    #[must_use]
    pub fn has_open_tool_span(&self, tool_name: &str, trace_id: &str) -> bool {
        lock(&self.spans).values().any(|state| {
            state.from_tool_rail
                && state.attribute(GEN_AI_TOOL_NAME) == Some(tool_name)
                && state.attribute(GEN_AI_OPERATION_NAME) == Some(GEN_AI_OPERATION_EXECUTE_TOOL)
                && (trace_id.is_empty() || state.attribute(OJ_TRACE_ID) == Some(trace_id))
        })
    }

    /// Derives the parent [`Context`] for a new span.
    ///
    /// Python counterpart: `_get_parent_context`, which calls
    /// `trace.set_span_in_context(state.span)`. `opentelemetry` 0.31 removed
    /// `set_span_in_context` and its contexts own their span, so the parent is
    /// rebuilt from the open span's [`SpanContext`] — the tracer only needs the
    /// parent's identity to link the child, and the child inherits the trace id
    /// and parent span id exactly as with a locally attached parent.
    ///
    /// Returns `None` when there is no parent id or no open span for it, which
    /// makes the new span a trace root.
    #[must_use]
    pub fn parent_context(&self, parent_invoke_id: Option<&str>) -> Option<Context> {
        let parent_invoke_id = parent_invoke_id?;
        let spans = lock(&self.spans);
        let parent = spans.get(parent_invoke_id)?;
        Some(Context::new().with_remote_span_context(parent.span_context()))
    }

    /// Returns the number of open spans.
    #[must_use]
    pub fn len(&self) -> usize {
        lock(&self.spans).len()
    }

    /// Reports whether no span is open.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        lock(&self.spans).is_empty()
    }
}

/// Per-invoke maps and incremental buffers of the workflow handler.
///
/// Kept in one struct so a lock covers the state and its three buffers
/// together: `push` and `pop` stay atomic with respect to the `append_*` calls,
/// and no method ever needs two locks.
#[derive(Debug, Default)]
struct WorkflowSpans {
    /// Open spans, keyed by invoke id.
    spans: BTreeMap<String, OtelSpanState>,
    /// Buffered `on_invoke_data` payloads, flushed as one span attribute.
    on_invoke_data: BTreeMap<String, Vec<Value>>,
    /// Buffered stream-input chunks, flushed as one span attribute.
    stream_inputs: BTreeMap<String, Vec<Value>>,
    /// Buffered stream-output chunks, flushed as one span attribute.
    stream_outputs: BTreeMap<String, Vec<Value>>,
}

impl WorkflowSpans {
    /// Appends to one buffer when the invoke id has a recorded open span.
    ///
    /// Python parity: a missing state or a non-recording span drops the chunk
    /// instead of creating a buffer.
    fn append(
        buffers: &mut BTreeMap<String, Vec<Value>>,
        spans: &BTreeMap<String, OtelSpanState>,
        invoke_id: &str,
        chunk: Value,
    ) {
        if !spans.get(invoke_id).is_some_and(|state| state.recorded) {
            return;
        }
        if let Some(buffer) = buffers.get_mut(invoke_id) {
            buffer.push(chunk);
        }
    }

    /// Returns a copy of one buffer, or an empty vector when unknown.
    fn snapshot(buffers: &BTreeMap<String, Vec<Value>>, invoke_id: &str) -> Vec<Value> {
        buffers.get(invoke_id).cloned().unwrap_or_default()
    }
}

/// Invoke id → [`OtelSpanState`] plus the workflow handler's incremental
/// buffers.
///
/// Python counterpart: `OtelWorkflowSpanManager`. Buffered data is flushed as a
/// single span attribute per buffer on `on_call_done`, which keeps span payloads
/// small instead of emitting one event per chunk.
#[derive(Debug, Default)]
pub struct OtelWorkflowSpanManager {
    /// Open spans and their buffers.
    inner: Mutex<WorkflowSpans>,
}

impl OtelWorkflowSpanManager {
    /// Creates an empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers an open span and resets its three buffers.
    pub fn push(&self, invoke_id: impl Into<String>, state: OtelSpanState) {
        let invoke_id = invoke_id.into();
        let mut inner = lock(&self.inner);
        inner.on_invoke_data.insert(invoke_id.clone(), Vec::new());
        inner.stream_inputs.insert(invoke_id.clone(), Vec::new());
        inner.stream_outputs.insert(invoke_id.clone(), Vec::new());
        inner.spans.insert(invoke_id, state);
    }

    /// Removes and returns a state together with its buffers.
    ///
    /// The buffers are dropped even when the state is already gone, so a
    /// half-cleaned invoke id cannot keep buffering.
    #[must_use]
    pub fn pop(&self, invoke_id: &str) -> Option<OtelSpanState> {
        let mut inner = lock(&self.inner);
        inner.on_invoke_data.remove(invoke_id);
        inner.stream_inputs.remove(invoke_id);
        inner.stream_outputs.remove(invoke_id);
        inner.spans.remove(invoke_id)
    }

    /// Runs `f` against the open state of an invoke id.
    ///
    /// Returns `None` when the invoke id is unknown. `f` runs under the manager
    /// lock and must not call back into the same manager.
    pub fn with_state<R>(
        &self,
        invoke_id: &str,
        f: impl FnOnce(&mut OtelSpanState) -> R,
    ) -> Option<R> {
        let mut inner = lock(&self.inner);
        inner.spans.get_mut(invoke_id).map(f)
    }

    /// Reports whether an invoke id still has an open span.
    #[must_use]
    pub fn contains(&self, invoke_id: &str) -> bool {
        lock(&self.inner).spans.contains_key(invoke_id)
    }

    /// Buffers one `on_invoke_data` payload.
    ///
    /// Dropped when the invoke id has no open state or the span does not
    /// record.
    pub fn append_on_invoke_data(&self, invoke_id: &str, data: Value) {
        let mut inner = lock(&self.inner);
        let WorkflowSpans {
            spans,
            on_invoke_data,
            ..
        } = &mut *inner;
        WorkflowSpans::append(on_invoke_data, spans, invoke_id, data);
    }

    /// Returns the buffered `on_invoke_data` payloads of an invoke id.
    #[must_use]
    pub fn on_invoke_data(&self, invoke_id: &str) -> Vec<Value> {
        let inner = lock(&self.inner);
        WorkflowSpans::snapshot(&inner.on_invoke_data, invoke_id)
    }

    /// Buffers one stream-input chunk.
    ///
    /// Dropped when the invoke id has no open state or the span does not
    /// record.
    pub fn append_stream_input(&self, invoke_id: &str, chunk: Value) {
        let mut inner = lock(&self.inner);
        let WorkflowSpans {
            spans,
            stream_inputs,
            ..
        } = &mut *inner;
        WorkflowSpans::append(stream_inputs, spans, invoke_id, chunk);
    }

    /// Returns the buffered stream-input chunks of an invoke id.
    #[must_use]
    pub fn stream_inputs(&self, invoke_id: &str) -> Vec<Value> {
        let inner = lock(&self.inner);
        WorkflowSpans::snapshot(&inner.stream_inputs, invoke_id)
    }

    /// Buffers one stream-output chunk.
    ///
    /// Dropped when the invoke id has no open state or the span does not
    /// record.
    pub fn append_stream_output(&self, invoke_id: &str, chunk: Value) {
        let mut inner = lock(&self.inner);
        let WorkflowSpans {
            spans,
            stream_outputs,
            ..
        } = &mut *inner;
        WorkflowSpans::append(stream_outputs, spans, invoke_id, chunk);
    }

    /// Returns the buffered stream-output chunks of an invoke id.
    #[must_use]
    pub fn stream_outputs(&self, invoke_id: &str) -> Vec<Value> {
        let inner = lock(&self.inner);
        WorkflowSpans::snapshot(&inner.stream_outputs, invoke_id)
    }

    /// Returns the number of open spans.
    #[must_use]
    pub fn len(&self) -> usize {
        lock(&self.inner).spans.len()
    }

    /// Reports whether no span is open.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        lock(&self.inner).spans.is_empty()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::otel::config::{ExporterType, OtelTracerConfig};
    use crate::otel::handle::{init_otel_tracer, TracerHandle};
    use opentelemetry::trace::Tracer as _;
    use serde_json::json;

    // port of tests/unit_tests/extensions/tracer_otel/test_span_manager.py and
    // the span-manager assertions of test_otel_tool_span_dedup.py.

    /// A tracer over a private in-memory provider.
    fn tracer_handle() -> TracerHandle {
        let config = OtelTracerConfig::default().with_exporter_type(ExporterType::Memory);
        init_otel_tracer(config).expect("the memory exporter always initializes")
    }

    /// Starts a recording root span.
    fn start_span(handle: &TracerHandle, name: &'static str) -> Span {
        handle
            .tracer()
            .expect("the memory handle is active")
            .start_with_context(name, &Context::new())
    }

    /// Asserts a workflow buffer holds nothing.
    ///
    /// A dedicated helper keeps the assertion style in one place (`clippy`
    /// prefers a comparison over `assert!(…is_empty())`).
    fn assert_empty(values: &[Value], buffer: &str) {
        assert_eq!(values, [] as [Value; 0], "{buffer} must be empty");
    }

    /// Starts a span and wraps it in a state with the given recorded flag.
    fn state(
        handle: &TracerHandle,
        invoke_id: &str,
        name: &'static str,
        recorded: bool,
    ) -> OtelSpanState {
        OtelSpanState::new(start_span(handle, name), invoke_id, recorded)
    }

    /// Builds a rail tool span state carrying the tool identity attributes.
    fn tool_state(
        handle: &TracerHandle,
        invoke_id: &str,
        tool_name: &str,
        trace_id: &str,
        from_tool_rail: bool,
    ) -> OtelSpanState {
        let mut state =
            OtelSpanState::from_span(start_span(handle, "execute_tool get_weather"), invoke_id);
        state.from_tool_rail = from_tool_rail;
        state.record_attribute(GEN_AI_TOOL_NAME, tool_name);
        state.record_attribute(GEN_AI_OPERATION_NAME, GEN_AI_OPERATION_EXECUTE_TOOL);
        if !trace_id.is_empty() {
            state.record_attribute(OJ_TRACE_ID, trace_id);
        }
        state
    }

    #[test]
    fn agent_manager_pushes_pops_and_counts() {
        let handle = tracer_handle();
        let manager = OtelAgentSpanManager::new();
        assert!(manager.is_empty());
        manager.push("inv1", state(&handle, "inv1", "chat Agent", true));
        assert!(manager.contains("inv1"));
        assert_eq!(manager.len(), 1);
        let seen = manager.with_state("inv1", |state| state.invoke_id.clone());
        assert_eq!(seen, Some("inv1".to_owned()));
        assert_eq!(manager.with_state("unknown", |_| ()), None);
        let popped = manager.pop("inv1").expect("the state is stored");
        assert_eq!(popped.invoke_id, "inv1");
        assert!(!manager.contains("inv1"));
        assert!(manager.is_empty());
        assert!(manager.pop("inv1").is_none());
    }

    #[test]
    fn agent_manager_replaces_a_state_under_the_same_invoke_id() {
        let handle = tracer_handle();
        let manager = OtelAgentSpanManager::new();
        manager.push("inv1", state(&handle, "inv1", "first", true));
        manager.push("inv1", state(&handle, "inv1", "second", true));
        assert_eq!(manager.len(), 1);
    }

    #[test]
    fn pop_returns_the_state_so_the_caller_can_end_the_span() {
        let handle = tracer_handle();
        let manager = OtelAgentSpanManager::new();
        manager.push("inv1", state(&handle, "inv1", "chat Agent", true));
        let mut popped = manager.pop("inv1").expect("the state is stored");
        popped.span.end();
        assert!(popped.span.exported_data().is_none(), "the span is closed");
    }

    #[test]
    fn from_span_derives_the_recorded_flag_from_a_dropped_span() {
        // A zero sample rate drops the span, so it records nothing (Python's
        // `recorded = span.is_recording()`).
        let config = OtelTracerConfig {
            sample_rate: 0.0,
            ..OtelTracerConfig::default()
        }
        .with_exporter_type(ExporterType::Memory);
        let handle = init_otel_tracer(config).expect("the memory exporter always initializes");
        let state = OtelSpanState::from_span(start_span(&handle, "chat Agent"), "inv1");
        assert!(!state.recorded, "a dropped span does not record");
        assert!(!state.span.is_recording());
    }

    #[test]
    fn state_reports_the_span_identity() {
        let handle = tracer_handle();
        let mut state = OtelSpanState::from_span(start_span(&handle, "chat Agent"), "inv1");
        let context = state.span_context();
        assert!(context.is_valid());
        assert_eq!(context, *state.span.span_context());
        assert!(state.recorded, "a sampled in-memory span records");
        state.span.end();
    }

    #[test]
    fn record_attribute_sets_the_span_and_the_cache() {
        let handle = tracer_handle();
        let mut state = state(&handle, "inv1", "execute_tool get_weather", true);
        state.record_attribute(GEN_AI_TOOL_NAME, "get_weather");
        assert_eq!(state.attribute(GEN_AI_TOOL_NAME), Some("get_weather"));
        assert_eq!(state.attribute(OJ_TRACE_ID), None);
        let data = state.span.exported_data().expect("the span records");
        assert!(
            data.attributes
                .iter()
                .any(|pair| pair.key.as_str() == GEN_AI_TOOL_NAME
                    && pair.value.as_str() == "get_weather"),
            "the attribute reaches the span as well"
        );
        state.span.end();
    }

    #[test]
    fn state_new_captures_the_caller_supplied_recorded_flag() {
        let handle = tracer_handle();
        let mut state = state(&handle, "inv1", "chat Agent", false);
        assert!(!state.recorded);
        state.from_tool_rail = true;
        state.inputs = Some(json!({"query": "x"}));
        state.workflow_id = Some("wf-1".to_owned());
        assert_eq!(state.workflow_id.as_deref(), Some("wf-1"));
        assert!(state.start_time <= SystemTime::now());
        state.span.end();
    }

    #[test]
    fn debug_hides_the_span_and_names_the_bookkeeping_fields() {
        let handle = tracer_handle();
        let state = state(&handle, "inv1", "chat Agent", true);
        let rendered = format!("{state:?}");
        assert!(rendered.contains("OtelSpanState"), "{rendered}");
        assert!(rendered.contains("inv1"), "{rendered}");
        assert!(rendered.contains("recorded"), "{rendered}");
        assert!(!rendered.contains("SpanData"), "{rendered}");
    }

    #[test]
    fn parent_context_links_the_child_to_the_open_parent() {
        let handle = tracer_handle();
        let manager = OtelAgentSpanManager::new();
        manager.push("parent-1", state(&handle, "parent-1", "chat Parent", true));
        let parent_context = manager
            .with_state("parent-1", |state| state.span_context())
            .expect("the parent is open");
        let context = manager
            .parent_context(Some("parent-1"))
            .expect("the parent is open");
        let tracer = handle.tracer().expect("the memory handle is active");
        let mut child = tracer.start_with_context("chat Child", &context);
        let data = child.exported_data().expect("the child records");
        assert_eq!(data.span_context.trace_id(), parent_context.trace_id());
        assert_eq!(data.parent_span_id, parent_context.span_id());
        child.end();
    }

    #[test]
    fn parent_context_is_none_for_unknown_or_absent_parents() {
        let handle = tracer_handle();
        let manager = OtelAgentSpanManager::new();
        assert!(manager.parent_context(None).is_none());
        assert!(manager.parent_context(Some("missing")).is_none());
        manager.push("parent-1", state(&handle, "parent-1", "chat Parent", true));
        assert!(manager.parent_context(Some("parent-1")).is_some());
        // After the parent closes, no child can link onto it.
        let _ = manager.pop("parent-1");
        assert!(manager.parent_context(Some("parent-1")).is_none());
    }

    #[test]
    fn tool_dedup_requires_a_rail_flagged_execute_tool_span() {
        let handle = tracer_handle();
        let manager = OtelAgentSpanManager::new();
        // Rail span without gen_ai.tool.call.id — still matched (bug #2046).
        manager.push(
            "inv-rail",
            tool_state(&handle, "inv-rail", "get_weather", "trace-1", true),
        );
        assert!(manager.has_open_tool_span("get_weather", "trace-1"));
        // An empty trace id matches any trace, like Python's falsy check.
        assert!(manager.has_open_tool_span("get_weather", ""));
        // Tool name and trace id still scope the match.
        assert!(!manager.has_open_tool_span("other_tool", "trace-1"));
        assert!(!manager.has_open_tool_span("get_weather", "trace-2"));
        // An unflagged span (legacy decorator path) never matches.
        manager.push(
            "inv-deco",
            tool_state(&handle, "inv-deco", "get_weather", "trace-1", false),
        );
        assert!(manager.has_open_tool_span("get_weather", "trace-1"));
        let _ = manager.pop("inv-rail");
        assert!(!manager.has_open_tool_span("get_weather", "trace-1"));
    }

    #[test]
    fn tool_dedup_ignores_spans_without_the_execute_tool_operation() {
        let handle = tracer_handle();
        let manager = OtelAgentSpanManager::new();
        let mut state = tool_state(&handle, "inv-chat", "get_weather", "trace-1", true);
        state.record_attribute(GEN_AI_OPERATION_NAME, "chat");
        manager.push("inv-chat", state);
        assert!(
            !manager.has_open_tool_span("get_weather", "trace-1"),
            "a chat span must not suppress a tool span"
        );
    }

    #[test]
    fn tool_dedup_ignores_tool_spans_without_a_cached_name() {
        let handle = tracer_handle();
        let manager = OtelAgentSpanManager::new();
        let mut state = OtelSpanState::from_span(start_span(&handle, "execute_tool x"), "inv-1");
        state.from_tool_rail = true;
        state.record_attribute(GEN_AI_OPERATION_NAME, GEN_AI_OPERATION_EXECUTE_TOOL);
        manager.push("inv-1", state);
        assert!(!manager.has_open_tool_span("get_weather", "trace-1"));
        assert!(!manager.has_open_tool_span("", "trace-1"));
    }

    #[test]
    fn workflow_manager_pushes_pops_and_counts() {
        let handle = tracer_handle();
        let manager = OtelWorkflowSpanManager::new();
        assert!(manager.is_empty());
        manager.push("inv1", state(&handle, "inv1", "chat wf", true));
        assert!(manager.contains("inv1"));
        assert_eq!(manager.len(), 1);
        assert_eq!(
            manager.with_state("inv1", |state| state.invoke_id.clone()),
            Some("inv1".to_owned())
        );
        assert_eq!(manager.with_state("missing", |_| ()), None);
        assert!(manager.pop("inv1").is_some());
        assert!(!manager.contains("inv1"));
        assert!(manager.is_empty());
        assert!(manager.pop("inv1").is_none());
    }

    #[test]
    fn workflow_pop_clears_every_buffer() {
        let handle = tracer_handle();
        let manager = OtelWorkflowSpanManager::new();
        manager.push("inv1", state(&handle, "inv1", "chat wf", true));
        manager.append_on_invoke_data("inv1", json!({"step": 1}));
        manager.append_stream_input("inv1", json!({"chunk": "a"}));
        manager.append_stream_output("inv1", json!({"chunk": "b"}));
        assert!(manager.pop("inv1").is_some());
        assert_empty(&manager.on_invoke_data("inv1"), "on_invoke_data");
        assert_empty(&manager.stream_inputs("inv1"), "stream_inputs");
        assert_empty(&manager.stream_outputs("inv1"), "stream_outputs");
    }

    #[test]
    fn workflow_buffers_keep_the_append_order() {
        let handle = tracer_handle();
        let manager = OtelWorkflowSpanManager::new();
        manager.push("inv1", state(&handle, "inv1", "chat wf", true));
        manager.append_on_invoke_data("inv1", json!({"step": 1}));
        manager.append_on_invoke_data("inv1", json!({"step": 2}));
        manager.append_stream_input("inv1", json!({"text": "hello"}));
        manager.append_stream_input("inv1", json!({"text": "world"}));
        manager.append_stream_output("inv1", json!({"result": "ok"}));
        assert_eq!(
            manager.on_invoke_data("inv1"),
            vec![json!({"step": 1}), json!({"step": 2})]
        );
        assert_eq!(
            manager.stream_inputs("inv1"),
            vec![json!({"text": "hello"}), json!({"text": "world"})]
        );
        assert_eq!(
            manager.stream_outputs("inv1"),
            vec![json!({"result": "ok"})]
        );
    }

    #[test]
    fn workflow_buffers_start_empty_and_are_empty_for_unknown_invokes() {
        let handle = tracer_handle();
        let manager = OtelWorkflowSpanManager::new();
        manager.push("inv1", state(&handle, "inv1", "chat wf", true));
        assert_empty(&manager.on_invoke_data("inv1"), "on_invoke_data");
        assert_empty(&manager.stream_inputs("inv1"), "stream_inputs");
        assert_empty(&manager.stream_outputs("inv1"), "stream_outputs");
        assert_empty(&manager.on_invoke_data("missing"), "on_invoke_data");
        assert_empty(&manager.stream_inputs("missing"), "stream_inputs");
        assert_empty(&manager.stream_outputs("missing"), "stream_outputs");
    }

    #[test]
    fn workflow_buffers_drop_chunks_without_state_or_when_not_recorded() {
        let handle = tracer_handle();
        let manager = OtelWorkflowSpanManager::new();
        // No state at all.
        manager.append_on_invoke_data("missing", json!({"step": 1}));
        manager.append_stream_input("missing", json!({"text": "x"}));
        manager.append_stream_output("missing", json!({"result": "y"}));
        assert_empty(&manager.on_invoke_data("missing"), "on_invoke_data");
        assert_empty(&manager.stream_inputs("missing"), "stream_inputs");
        assert_empty(&manager.stream_outputs("missing"), "stream_outputs");
        // A non-recording span buffers nothing.
        manager.push("inv2", state(&handle, "inv2", "chat wf", false));
        manager.append_on_invoke_data("inv2", json!({"step": 1}));
        manager.append_stream_input("inv2", json!({"text": "x"}));
        manager.append_stream_output("inv2", json!({"result": "y"}));
        assert_empty(&manager.on_invoke_data("inv2"), "on_invoke_data");
        assert_empty(&manager.stream_inputs("inv2"), "stream_inputs");
        assert_empty(&manager.stream_outputs("inv2"), "stream_outputs");
    }

    #[test]
    fn managers_are_shareable_across_threads() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<OtelAgentSpanManager>();
        assert_send_sync::<OtelWorkflowSpanManager>();
        let handle = tracer_handle();
        let manager = std::sync::Arc::new(OtelAgentSpanManager::new());
        manager.push("inv1", state(&handle, "inv1", "chat Agent", true));
        let worker = std::sync::Arc::clone(&manager);
        let seen =
            std::thread::spawn(move || worker.with_state("inv1", |state| state.invoke_id.clone()))
                .join()
                .expect("the worker thread finishes");
        assert_eq!(seen, Some("inv1".to_owned()));
    }
}
