//! Span references, handler state, and the shared attribute writers.
//!
//! Split out of `workflow_handler/mod.rs` to keep both files readable: this
//! file owns the lightweight [`SpanRef`] bookkeeping the handler keeps instead
//! of Python's shared `OtelSpanState` objects, the single [`HandlerState`] that
//! sits behind the handler mutex, the per-call [`CallFacts`] classifier, and the
//! attribute writers ported from `_set_workflow_attrs`,
//! `_set_workflow_end_attrs`, `_is_retrieval_component_span`,
//! `_extract_retrieval_query`, and `_extract_retrieval_documents`.
//!
//! Every item is crate-internal; the public surface lives in `mod.rs`.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Mutex, MutexGuard};
use std::time::SystemTime;

use opentelemetry::trace::{Span as _, SpanContext, SpanKind, TraceContextExt as _};
use opentelemetry::{Array, Context, KeyValue, StringValue, Value as OtelValue};
use serde_json::Value;

use super::super::attrs;
use super::super::redaction::Redactor;
use super::super::semconv::{
    GEN_AI_OPERATION_RETRIEVAL, GEN_AI_RETRIEVAL_DOCUMENTS, OJ_ELAPSED_TIME, OJ_END_TIME,
    OJ_WORKFLOW_COMPONENT_ID, OJ_WORKFLOW_COMPONENT_NAME, OJ_WORKFLOW_COMPONENT_TYPE,
    OJ_WORKFLOW_EXECUTION_ID, OJ_WORKFLOW_ID, OJ_WORKFLOW_LOOP_INDEX, OJ_WORKFLOW_LOOP_NODE_ID,
    OJ_WORKFLOW_NAME, OJ_WORKFLOW_VERSION,
};
use super::super::span_manager::OtelSpanState;
use super::super::span_names;

/// Locks the handler mutex, recovering the data when a previous holder panicked.
///
/// Tracing is fail-open: a poisoned lock must not turn a trace callback into a
/// panic, so the state is used as-is instead of propagating the poison.
pub(super) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Reference to an open span held by the layer-root and component maps.
///
/// Python keeps the whole `OtelSpanState` in those maps; the Rust state owns its
/// span and is not `Clone`, so the handler keeps the span identity instead and
/// resolves the live state through `OtelWorkflowSpanManager`.
#[derive(Debug, Clone)]
pub(super) struct SpanRef {
    /// Invoke id the span is registered under.
    pub(super) invoke_id: String,
    /// Span identity, cloned at start; used to parent child spans.
    pub(super) span_context: SpanContext,
    /// Owning workflow id, used for cross-workflow isolation.
    pub(super) workflow_id: Option<String>,
    /// Registration order, so "the last component span" is well defined.
    pub(super) sequence: u64,
}

impl SpanRef {
    /// Derives the parent [`Context`] of a new span from this span.
    ///
    /// Python counterpart: `_get_parent_context`, which calls
    /// `trace.set_span_in_context(state.span)`. `opentelemetry` 0.31 has no
    /// `set_span_in_context`, so the context carries the span's identity as a
    /// remote parent; the child inherits the trace id and parent span id.
    pub(super) fn parent_context(&self) -> Context {
        Context::new().with_remote_span_context(self.span_context.clone())
    }
}

/// Mutable handler state, kept behind the handler's single mutex.
///
/// Python keeps these fields on the handler instance and relies on one async
/// context running the callbacks sequentially. The Rust handler is `Send +
/// Sync`, so the fields live behind one lock; no lock is ever held across an
/// `await`, and a closure passed to the span manager must never lock this state
/// (the manager's lock is taken first there).
#[derive(Default)]
pub(super) struct HandlerState {
    /// Trace id written to `openjiuwen.trace.id`.
    pub(super) trace_id: String,
    /// Session id used when a frame carries none.
    pub(super) session_id: Option<String>,
    /// Layer roots: `parent_node_id` → workflow root span.
    pub(super) layer_roots: BTreeMap<String, SpanRef>,
    /// Component spans: invoke id → host component span.
    pub(super) components: BTreeMap<String, SpanRef>,
    /// Root context cached for the next conversation round.
    pub(super) cached_root_ctx: Option<Context>,
    /// Trace id the cached root context belongs to (bookkeeping, as in Python).
    pub(super) cached_trace_id: Option<String>,
    /// Session id of the last `set_session_id` call that cleaned the maps.
    pub(super) cached_session_id: Option<String>,
    /// Monotonic registration counter behind [`SpanRef::sequence`].
    pub(super) next_sequence: u64,
}

impl HandlerState {
    /// Allocates the registration sequence of a new span reference.
    pub(super) fn next_sequence(&mut self) -> u64 {
        let sequence = self.next_sequence;
        self.next_sequence += 1;
        sequence
    }

    /// Clears the cached root context.
    pub(super) fn clear_cached_context(&mut self) {
        self.cached_root_ctx = None;
        self.cached_trace_id = None;
    }

    /// Remembers the root context of `span`, which the next round reuses.
    pub(super) fn cache_root(&mut self, span: &SpanRef, trace_id: &str) {
        self.cached_root_ctx = Some(span.parent_context());
        self.cached_trace_id = Some(trace_id.to_owned());
    }

    /// The most recently registered component span.
    ///
    /// Python counterpart: `list(self._component_spans.values())[-1]`.
    fn last_component(&self) -> Option<&SpanRef> {
        self.components.values().max_by_key(|span| span.sequence)
    }

    /// Resolves the parent context of a new span.
    ///
    /// Python counterpart: `_resolve_parent_context`. A workflow root with an
    /// empty `parent_node_id` either restarts the trace (another workflow id),
    /// continues the cached root context (multi-round conversation), or parents
    /// onto the last component span / the existing layer root. A component with
    /// an empty `parent_node_id` parents onto the root workflow's layer root,
    /// and any other parent id is resolved through the component map first and
    /// the layer-root map second — the sub-workflow host component therefore
    /// wins over a layer root registered under the same key.
    ///
    /// The cross-workflow branch clears both maps without ending the spans,
    /// exactly like Python: those spans belong to a superseded execution.
    pub(super) fn resolve_parent_context(
        &mut self,
        parent_node_id: &str,
        metadata: &Value,
    ) -> Option<Context> {
        if parent_node_id.is_empty() && is_workflow_root(metadata) {
            let Some(existing) = self.layer_roots.get("").cloned() else {
                return self.cached_root_ctx.clone();
            };
            let workflow_id = text_field(metadata, "workflow_id").filter(|id| !id.is_empty());
            if let Some(workflow_id) = workflow_id {
                if existing.workflow_id.as_deref() != Some(workflow_id) {
                    self.layer_roots.clear();
                    self.components.clear();
                    return None;
                }
            }
            if let Some(component) = self.last_component() {
                return Some(component.parent_context());
            }
            return Some(existing.parent_context());
        }
        if parent_node_id.is_empty() {
            return self.layer_roots.get("").map(SpanRef::parent_context);
        }
        self.components
            .get(parent_node_id)
            .or_else(|| self.layer_roots.get(parent_node_id))
            .map(SpanRef::parent_context)
    }
}

impl fmt::Debug for HandlerState {
    /// Renders the bookkeeping fields; contexts and span internals stay private.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HandlerState")
            .field("trace_id", &self.trace_id)
            .field("session_id", &self.session_id)
            .field("layer_roots", &self.layer_roots.keys().collect::<Vec<_>>())
            .field("components", &self.components.keys().collect::<Vec<_>>())
            .field("has_cached_root_context", &self.cached_root_ctx.is_some())
            .finish_non_exhaustive()
    }
}

/// The span family a workflow invocation belongs to.
///
/// Python recomputes the three component-type classifiers inline and selects
/// the name/kind with an `if`/`elif` chain; keeping one enum makes the branches
/// mutually exclusive and keeps the fallbacks in a single place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SpanFamily {
    /// A workflow layer root: span name = the invoke id.
    Root,
    /// An LLM-bearing component: `chat {model}`, [`SpanKind::Client`].
    Llm,
    /// A tool-executing component: `execute_tool {name}`.
    Tool,
    /// A retrieval component: `retrieval {name}`.
    Retrieval,
    /// Any other component: `component.{invoke_id}`.
    Component,
}

/// Per-call facts derived from the `on_call_start` metadata.
///
/// Python reads the same fields inline in `on_call_start` (span name, span kind,
/// and attribute branches); collecting them once keeps the classifier rules and
/// their fallbacks in one place.
pub(super) struct CallFacts {
    /// `parent_node_id` folded out of the metadata (default `""`).
    pub(super) parent_node_id: String,
    /// Trace id of the handler at call start.
    pub(super) trace_id: String,
    /// Session id: the frame's own, else the handler's.
    pub(super) session_id: Option<String>,
    /// Owning workflow id.
    pub(super) workflow_id: Option<String>,
    /// Span family of this invocation.
    pub(super) family: SpanFamily,
    /// `component_name` when present and non-empty.
    component_name: Option<String>,
    /// `model_name`, else `model`.
    model: Option<String>,
}

impl CallFacts {
    /// Classifies a start frame from its metadata.
    ///
    /// `trace_id` is filled in by the handler, which owns the state.
    pub(super) fn from_metadata(metadata: &Value) -> Self {
        let component_type = text_field(metadata, "component_type").unwrap_or_default();
        let family = if is_workflow_root(metadata) {
            SpanFamily::Root
        } else if span_names::is_llm_component(component_type) {
            SpanFamily::Llm
        } else if span_names::is_tool_component(component_type) {
            SpanFamily::Tool
        } else if span_names::is_retrieval_component(component_type) {
            SpanFamily::Retrieval
        } else {
            SpanFamily::Component
        };
        Self {
            parent_node_id: text_field(metadata, "parent_node_id")
                .unwrap_or_default()
                .to_owned(),
            trace_id: String::new(),
            session_id: truthy_text(metadata.get("session_id")),
            workflow_id: text_field(metadata, "workflow_id").map(str::to_owned),
            family,
            component_name: truthy_text(metadata.get("component_name")),
            model: truthy_text(metadata.get("model_name"))
                .or_else(|| truthy_text(metadata.get("model"))),
        }
    }

    /// Whether these facts describe a workflow root rather than a component.
    pub(super) fn is_root(&self) -> bool {
        self.family == SpanFamily::Root
    }

    /// Whether these facts describe a retrieval component.
    pub(super) fn is_retrieval(&self) -> bool {
        self.family == SpanFamily::Retrieval
    }

    /// The span name of this invocation.
    ///
    /// Python counterpart: the name branches of `on_call_start` — the invoke id
    /// for a workflow root, `chat {model}` for an LLM component,
    /// `execute_tool {name}` for a tool component, `retrieval {name}` for a
    /// retrieval component, and `component.{invoke_id}` otherwise.
    pub(super) fn span_name(&self, invoke_id: &str) -> String {
        match self.family {
            SpanFamily::Root => invoke_id.to_owned(),
            SpanFamily::Llm => {
                span_names::chat_span_name_for_model(self.model.as_deref().unwrap_or_default())
            }
            SpanFamily::Tool => span_names::tool_span_name(self.name(invoke_id)),
            SpanFamily::Retrieval => retrieval_span_name(self.name(invoke_id)),
            SpanFamily::Component => span_names::workflow_component_span_name(invoke_id),
        }
    }

    /// The span kind: LLM components are clients, everything else is internal.
    pub(super) fn span_kind(&self) -> SpanKind {
        if self.family == SpanFamily::Llm {
            SpanKind::Client
        } else {
            SpanKind::Internal
        }
    }

    /// The `model_name`/`model` value of an LLM component.
    pub(super) fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    /// The component name used by the tool/retrieval branches: name or invoke id.
    pub(super) fn name<'a>(&'a self, invoke_id: &'a str) -> &'a str {
        self.component_name.as_deref().unwrap_or(invoke_id)
    }
}

impl fmt::Debug for CallFacts {
    /// Renders the classifier output; the payload strings stay out of `Debug`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CallFacts")
            .field("parent_node_id", &self.parent_node_id)
            .field("trace_id", &self.trace_id)
            .field("has_session_id", &self.session_id.is_some())
            .field("workflow_id", &self.workflow_id)
            .field("family", &self.family)
            .field("has_component_name", &self.component_name.is_some())
            .field("model", &self.model)
            .finish()
    }
}

/// Reads a string field of a metadata payload, ignoring non-object payloads.
pub(super) fn text_field<'a>(metadata: &'a Value, key: &str) -> Option<&'a str> {
    metadata.get(key).and_then(Value::as_str)
}

/// Reports whether a metadata payload carries a key.
fn has_key(metadata: &Value, key: &str) -> bool {
    metadata
        .as_object()
        .is_some_and(|map| map.contains_key(key))
}

/// Renders a metadata field the way Python's `str(value) if value else ""` does.
///
/// Strings are used verbatim and an empty string counts as absent; every other
/// non-null JSON value is rendered as text, so `12` becomes `"12"`.
pub(super) fn truthy_text(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::Null => None,
        Value::String(text) if text.is_empty() => None,
        Value::String(text) => Some(text.clone()),
        other => Some(attrs::serialize_value(other)),
    }
}

/// Whether metadata describes a workflow root.
///
/// Python counterpart: `"workflow_id" in metadata and "component_id" not in
/// metadata`, which is also the discriminator of `_set_workflow_attrs`.
fn is_workflow_root(metadata: &Value) -> bool {
    has_key(metadata, "workflow_id") && !has_key(metadata, "component_id")
}

/// Wraps strings in an `OTel` array attribute value.
pub(super) fn string_array(values: impl IntoIterator<Item = String>) -> OtelValue {
    OtelValue::Array(Array::String(
        values.into_iter().map(StringValue::from).collect(),
    ))
}

/// The span name of a retrieval component: `retrieval {name}`.
///
/// Python formats `f"retrieval {name}"`; the space-joined `GenAI` style keeps
/// the bare operation name when the component name is blank, the same rule as
/// [`span_names::tool_span_name`].
fn retrieval_span_name(name: &str) -> String {
    if name.trim().is_empty() {
        return GEN_AI_OPERATION_RETRIEVAL.to_owned();
    }
    format!("{GEN_AI_OPERATION_RETRIEVAL} {name}")
}

/// Pulls the query text out of a retrieval component's inputs.
///
/// Python counterpart: `_extract_retrieval_query` — a non-empty string is the
/// query, a non-empty `query` field of an object is the query, everything else
/// (including an empty query) yields `None`.
pub(super) fn retrieval_query(inputs: &Value) -> Option<String> {
    match inputs {
        Value::String(text) => (!text.is_empty()).then(|| text.clone()),
        other => text_field(other, "query")
            .filter(|query| !query.is_empty())
            .map(str::to_owned),
    }
}

/// Pulls the retrieved documents out of a retrieval component's outputs.
///
/// Python counterpart: `_extract_retrieval_documents` — the `results` array of
/// an object payload, rendered with `_serialize` per item; an absent, empty, or
/// non-array result yields `None`.
fn retrieval_documents(outputs: &Value) -> Option<Vec<String>> {
    let results = outputs.get("results")?.as_array()?;
    if results.is_empty() {
        return None;
    }
    Some(results.iter().map(attrs::serialize_value).collect())
}

/// Whether an open component state belongs to a retrieval component.
///
/// Python counterpart: `_is_retrieval_component_span`, which reads the span's
/// `openjiuwen.workflow.component.type`. The SDK span does not expose its
/// attributes, so the value comes from [`OtelSpanState`]'s cache — which is why
/// the start path records that key through
/// [`OtelSpanState::record_attribute`].
pub(super) fn is_retrieval_state(state: &OtelSpanState) -> bool {
    state
        .attribute(OJ_WORKFLOW_COMPONENT_TYPE)
        .is_some_and(span_names::is_retrieval_component)
}

/// Writes the `openjiuwen.workflow.*` attributes of a root or component span.
///
/// Python counterpart: `_set_workflow_attrs`. A root records its workflow
/// identity (execution id defaulting to the invoke id); a component records its
/// own identity, the owning workflow id, and the loop coordinates when present.
pub(super) fn set_workflow_attributes(
    state: &mut OtelSpanState,
    metadata: &Value,
    invoke_id: &str,
) {
    if metadata.is_null() {
        return;
    }
    if is_workflow_root(metadata) {
        state.record_attribute(
            OJ_WORKFLOW_ID,
            text_field(metadata, "workflow_id").unwrap_or_default(),
        );
        state.record_attribute(
            OJ_WORKFLOW_NAME,
            text_field(metadata, "workflow_name").unwrap_or_default(),
        );
        state.record_attribute(
            OJ_WORKFLOW_VERSION,
            text_field(metadata, "workflow_version").unwrap_or_default(),
        );
        state.record_attribute(
            OJ_WORKFLOW_EXECUTION_ID,
            text_field(metadata, "workflow_id").unwrap_or(invoke_id),
        );
        return;
    }
    state.record_attribute(
        OJ_WORKFLOW_COMPONENT_ID,
        text_field(metadata, "component_id").unwrap_or_default(),
    );
    state.record_attribute(
        OJ_WORKFLOW_COMPONENT_TYPE,
        text_field(metadata, "component_type").unwrap_or_default(),
    );
    state.record_attribute(
        OJ_WORKFLOW_COMPONENT_NAME,
        text_field(metadata, "component_name").unwrap_or_default(),
    );
    state.record_attribute(
        OJ_WORKFLOW_ID,
        text_field(metadata, "workflow_id").unwrap_or_default(),
    );
    if let Some(loop_node_id) = metadata
        .get("loop_node_id")
        .filter(|value| !value.is_null())
    {
        state.record_attribute(
            OJ_WORKFLOW_LOOP_NODE_ID,
            attrs::serialize_value(loop_node_id),
        );
    }
    if let Some(loop_index) = metadata.get("loop_index").filter(|value| !value.is_null()) {
        state.record_attribute(OJ_WORKFLOW_LOOP_INDEX, attrs::serialize_value(loop_index));
    }
}

/// Writes the closing attributes of a workflow span.
///
/// Python counterpart: `_set_workflow_end_attrs` — the local end time and the
/// elapsed time derived from the state's cached start time, in the same two
/// shapes [`attrs::elapsed_string`] produces.
pub(super) fn set_end_attributes(state: &mut OtelSpanState) {
    let end_time = attrs::local_timestamp();
    attrs::set_text_attribute(&mut state.span, OJ_END_TIME, &end_time);
    let elapsed = SystemTime::now()
        .duration_since(state.start_time)
        .map_or(0, |elapsed| {
            u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX)
        });
    attrs::set_text_attribute(
        &mut state.span,
        OJ_ELAPSED_TIME,
        &attrs::elapsed_string(elapsed),
    );
}

/// Writes the redacted retrieval documents of `outputs` onto a span.
///
/// Python counterpart: the `_is_retrieval_component_span` /
/// `_extract_retrieval_documents` block of `on_call_done` and `on_post_invoke`.
/// Documents follow the same policy as the query text and the workflow outputs:
/// hashed under redaction, truncated otherwise — never raw document text.
pub(super) fn record_documents(state: &mut OtelSpanState, outputs: &Value, redactor: Redactor<'_>) {
    let Some(documents) = retrieval_documents(outputs) else {
        return;
    };
    let documents = documents
        .iter()
        .map(|document| redactor.redact(Some(&Value::String(document.clone())), None))
        .collect::<Vec<_>>();
    state.span.set_attribute(KeyValue::new(
        GEN_AI_RETRIEVAL_DOCUMENTS,
        string_array(documents),
    ));
}
