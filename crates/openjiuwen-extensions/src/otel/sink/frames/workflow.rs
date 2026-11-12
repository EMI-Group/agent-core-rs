//! Workflow and component frame mapping for [`super::OtelTraceSink`].
//!
//! Split out of `frames.rs` to keep both files readable: this file owns the
//! port of `tracer_otel/handler.py`'s workflow handler — the component
//! classifier ([`WorkflowFamily`]), the `on_call_*`/`on_invoke` mapping, the
//! `meta_data` buffers, and the lazy open of a span from its first
//! `running`/terminal frame.
//!
//! The helpers shared with the agent path stay in [`super`]: this module never
//! re-types an attribute key or a serialization rule.
use std::time::SystemTime;

use opentelemetry::trace::{Span as _, SpanKind, TraceContextExt as _, Tracer as _};
use opentelemetry::{Context, KeyValue};
use serde_json::Value;

use openjiuwen_core::session::tracer::{SpanStatus, TraceSpan, TraceSpanKind};

use super::{
    component_entity, end_span, frame_time, is_retrieval_state, json_i64, meta_text,
    metadata_value, model_name, present, retrieval_span_name, serialize_buffer, span_context_of,
    string_array, truthy_text, write_component_identity, write_retrieval_documents, LOG_TARGET,
};
use crate::otel::attrs::{self, AttributeScope};
use crate::otel::llm_attrs::extract_retrieval_query;
use crate::otel::redaction::Redactor;
use crate::otel::semconv::{
    GEN_AI_DATA_SOURCE_ID, GEN_AI_OPERATION_CHAT, GEN_AI_OPERATION_EXECUTE_TOOL,
    GEN_AI_OPERATION_NAME, GEN_AI_OPERATION_RETRIEVAL, GEN_AI_REQUEST_MODEL,
    GEN_AI_RETRIEVAL_QUERY_TEXT, GEN_AI_RETRIEVAL_TOP_K, GEN_AI_TOOL_NAME, OJ_META_DATA,
    OJ_PARENT_NODE_ID, OJ_SOURCE_IDS, OJ_STATUS, OJ_STREAM_INPUTS, OJ_STREAM_OUTPUTS,
    OJ_WORKFLOW_EXECUTION_ID, OJ_WORKFLOW_ID, OJ_WORKFLOW_INPUTS, OJ_WORKFLOW_INVOKE_DATA,
    OJ_WORKFLOW_NAME, OJ_WORKFLOW_OUTPUTS, OJ_WORKFLOW_VERSION, SPAN_STATUS_RUNNING,
};
use crate::otel::span_manager::OtelSpanState;
use crate::otel::span_names;

/// The three incremental buffers of the workflow handler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Buffer {
    /// `openjiuwen.workflow.invoke_data` (Python `on_invoke`).
    InvokeData,
    /// `openjiuwen.stream_inputs` (Python `on_pre_stream`).
    StreamInputs,
    /// `openjiuwen.stream_outputs` (Python `on_post_stream`).
    StreamOutputs,
}

/// Metadata keys that feed each buffer, in flush order.
const BUFFER_KEYS: [(&str, Buffer); 3] = [
    ("on_invoke_data", Buffer::InvokeData),
    ("stream_inputs", Buffer::StreamInputs),
    ("stream_outputs", Buffer::StreamOutputs),
];

/// The span family a workflow frame belongs to.
///
/// Python recomputes `is_llm_component` / `is_tool_component` /
/// `is_retrieval_component` inline and selects the name/kind with an
/// `if`/`elif` chain; one enum keeps the branches mutually exclusive and the
/// fallbacks in a single place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkflowFamily {
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

impl WorkflowFamily {
    /// Classifies a frame.
    ///
    /// Python counterpart: the `is_workflow_root` / component-type chain of
    /// `on_call_start`. The frame kind replaces `"workflow_id" in metadata`:
    /// [`TraceSpanKind::Workflow`] is a layer root, and every component frame
    /// carries its type in `component_type`.
    fn classify(payload: &TraceSpan) -> Self {
        if payload.kind == TraceSpanKind::Workflow {
            return Self::Root;
        }
        let component_type = payload.component_type.as_deref().unwrap_or_default();
        if span_names::is_llm_component(component_type) {
            Self::Llm
        } else if span_names::is_tool_component(component_type) {
            Self::Tool
        } else if span_names::is_retrieval_component(component_type) {
            Self::Retrieval
        } else {
            Self::Component
        }
    }

    /// The span name of this frame.
    ///
    /// Python counterpart: the name branches of `on_call_start`.
    fn span_name(self, payload: &TraceSpan) -> String {
        match self {
            Self::Root => payload.invoke_id.clone(),
            Self::Llm => span_names::chat_span_name_for_model(
                model_name(payload).as_deref().unwrap_or_default(),
            ),
            Self::Tool => span_names::tool_span_name(component_entity(payload)),
            Self::Retrieval => retrieval_span_name(component_entity(payload)),
            Self::Component => span_names::workflow_component_span_name(&payload.invoke_id),
        }
    }

    /// The span kind: LLM components are clients, everything else is internal.
    fn span_kind(self) -> SpanKind {
        match self {
            Self::Llm => SpanKind::Client,
            _ => SpanKind::Internal,
        }
    }
}
impl super::super::OtelTraceSink {
    // -----------------------------------------------------------------------
    // Workflow path — port of `OtelWorkflowHandler`
    // -----------------------------------------------------------------------

    /// Starts a workflow or component span.
    ///
    /// Python counterpart: `on_call_start`. The span is registered before the
    /// buffers are fed, because appending requires an open state.
    pub(super) fn start_workflow_frame(&self, payload: &TraceSpan) {
        if let Some(state) = self.new_workflow_state(payload, None) {
            self.workflow_spans.push(payload.invoke_id.as_str(), state);
            self.feed_buffers(payload);
        }
    }

    /// Opens the workflow span of a frame that arrives without a start frame.
    ///
    /// The session tracer emits exactly one frame per span — its terminal one —
    /// so the first workflow/component frame the sink sees for an invoke id is
    /// normally `running` or terminal. Opening the span here keeps the idiomatic
    /// `Tracer::with_sink(..)` + `start_workflow_span(..)` + `finish(..)`
    /// integration exporting a span, using the frame's own `start_time` as the
    /// `OTel` start timestamp. Buffers are *not* fed here: every caller feeds
    /// them once, after the span is open.
    ///
    /// Returns `false` when the frame cannot be mapped (no tracer).
    fn open_workflow_lazily(&self, payload: &TraceSpan) -> bool {
        let Some(state) = self.new_workflow_state(payload, frame_time(payload.start_time)) else {
            return false;
        };
        self.workflow_spans.push(payload.invoke_id.as_str(), state);
        true
    }

    /// Builds the state of a workflow span without registering it.
    ///
    /// `start_time` overrides the span start instant; `None` starts the span
    /// now, which is what a real start frame does.
    fn new_workflow_state(
        &self,
        payload: &TraceSpan,
        start_time: Option<SystemTime>,
    ) -> Option<OtelSpanState> {
        let tracer = self.tracer.as_ref()?;
        let family = WorkflowFamily::classify(payload);
        let name = family.span_name(payload);
        let parent = self.workflow_parent_context(payload);
        let builder = tracer.span_builder(name).with_kind(family.span_kind());
        let builder = match start_time {
            Some(start_time) => builder.with_start_time(start_time),
            None => builder,
        };
        let span = builder.start_with_context(tracer, &parent);
        let mut state = OtelSpanState::from_span(span, payload.invoke_id.as_str());
        state.workflow_id.clone_from(&payload.workflow_id);
        if state.recorded {
            self.write_workflow_start(&mut state, payload, family);
        }
        Some(state)
    }

    /// Writes the start attributes of a recording workflow span.
    ///
    /// Python counterpart: `on_call_start`'s base attributes, its `gen_ai.*`
    /// component branches, `_set_workflow_attrs`, and the workflow-inputs /
    /// retrieval-query block.
    fn write_workflow_start(
        &self,
        state: &mut OtelSpanState,
        payload: &TraceSpan,
        family: WorkflowFamily,
    ) {
        attrs::apply_start_attributes(&mut state.span, payload);
        self.write_session_fallback(state, payload);
        state.record_attribute(
            OJ_PARENT_NODE_ID,
            payload.parent_invoke_id.as_deref().unwrap_or_default(),
        );
        if let Some(source_ids) = present(payload.meta_data.get("source_ids")) {
            state.record_attribute(OJ_SOURCE_IDS, attrs::serialize_value(source_ids));
        }
        if let Some(workflow_id) = payload.workflow_id.as_deref() {
            state.record_attribute(OJ_WORKFLOW_ID, workflow_id);
        }
        Self::write_family_attributes(state, payload, family);
        if family == WorkflowFamily::Root {
            state.record_attribute(OJ_WORKFLOW_NAME, meta_text(payload, "workflow_name"));
            state.record_attribute(OJ_WORKFLOW_VERSION, meta_text(payload, "workflow_version"));
            let execution_id = payload
                .meta_data
                .get("workflow_id")
                .map_or_else(|| payload.invoke_id.clone(), attrs::serialize_value);
            state.record_attribute(OJ_WORKFLOW_EXECUTION_ID, execution_id);
        } else {
            write_component_identity(state, payload);
        }
        if !payload.meta_data.is_empty() {
            state.record_attribute(
                OJ_META_DATA,
                attrs::serialize_value(&metadata_value(payload)),
            );
        }
        self.write_workflow_inputs(state, payload, family);
    }

    /// Writes the `gen_ai.*` attributes of a workflow span.
    ///
    /// Python counterpart: the `is_llm_component` / `is_tool_component` /
    /// `is_retrieval_component` branches of `on_call_start`.
    fn write_family_attributes(
        state: &mut OtelSpanState,
        payload: &TraceSpan,
        family: WorkflowFamily,
    ) {
        match family {
            WorkflowFamily::Llm => {
                state.record_attribute(GEN_AI_OPERATION_NAME, GEN_AI_OPERATION_CHAT);
                if let Some(model) = model_name(payload) {
                    state.record_attribute(GEN_AI_REQUEST_MODEL, model);
                }
            }
            WorkflowFamily::Tool => {
                state.record_attribute(GEN_AI_OPERATION_NAME, GEN_AI_OPERATION_EXECUTE_TOOL);
                state.record_attribute(GEN_AI_TOOL_NAME, component_entity(payload));
            }
            WorkflowFamily::Retrieval => {
                state.record_attribute(GEN_AI_OPERATION_NAME, GEN_AI_OPERATION_RETRIEVAL);
                if let Some(top_k) = payload.meta_data.get("top_k").and_then(json_i64) {
                    state
                        .span
                        .set_attribute(KeyValue::new(GEN_AI_RETRIEVAL_TOP_K, top_k));
                }
                if let Some(ids) = payload
                    .meta_data
                    .get("data_source_ids")
                    .and_then(Value::as_array)
                    .filter(|ids| !ids.is_empty())
                {
                    let ids = ids.iter().map(attrs::serialize_value).collect::<Vec<_>>();
                    state
                        .span
                        .set_attribute(KeyValue::new(GEN_AI_DATA_SOURCE_ID, string_array(ids)));
                }
                if let Some(model) = truthy_text(payload.meta_data.get("request_model")) {
                    state.record_attribute(GEN_AI_REQUEST_MODEL, model);
                }
            }
            WorkflowFamily::Root | WorkflowFamily::Component => {}
        }
    }

    /// Writes the redacted inputs of a workflow span.
    ///
    /// Python counterpart: the `inputs is not None` block of `on_call_start` and
    /// the inputs branch of `on_pre_invoke`, including the retrieval query text.
    fn write_workflow_inputs(
        &self,
        state: &mut OtelSpanState,
        payload: &TraceSpan,
        family: WorkflowFamily,
    ) {
        let Some(inputs) = present(payload.inputs.as_ref()) else {
            return;
        };
        let redactor = Redactor::new(&self.config);
        state.record_attribute(OJ_WORKFLOW_INPUTS, redactor.redact(Some(inputs), None));
        state.inputs = Some(inputs.clone());
        if family == WorkflowFamily::Retrieval {
            if let Some(query) = extract_retrieval_query(inputs) {
                state.record_attribute(
                    GEN_AI_RETRIEVAL_QUERY_TEXT,
                    redactor.redact(Some(&Value::String(query)), None),
                );
            }
        }
    }

    /// Refreshes a running workflow invocation.
    ///
    /// Python counterpart: `on_pre_invoke` plus the three buffering callbacks.
    /// The status flips to `running`, a fresh payload rewrites the inputs, and
    /// buffered chunks accumulate for the terminal flush. A running frame that is
    /// the first frame of its invoke id opens the span (and keeps it open, so a
    /// later terminal frame closes it).
    pub(super) fn running_workflow(&self, payload: &TraceSpan) {
        let opened_now = if self.workflow_spans.contains(&payload.invoke_id) {
            false
        } else if self.open_workflow_lazily(payload) {
            true
        } else {
            tracing::debug!(
                target: LOG_TARGET,
                invoke_id = %payload.invoke_id,
                "otel sink: dropping a running workflow frame with no mappable span"
            );
            return;
        };
        self.feed_buffers(payload);
        let redactor = Redactor::new(&self.config);
        let updated = self.workflow_spans.with_state(&payload.invoke_id, |state| {
            state.record_attribute(OJ_STATUS, SPAN_STATUS_RUNNING);
            // A lazily opened span already carries this frame's inputs through
            // its start attributes, so refreshing them would duplicate the write.
            if opened_now {
                return;
            }
            let Some(inputs) = present(payload.inputs.as_ref()) else {
                return;
            };
            state.record_attribute(OJ_WORKFLOW_INPUTS, redactor.redact(Some(inputs), None));
            state.inputs = Some(inputs.clone());
            if is_retrieval_state(state) {
                if let Some(query) = extract_retrieval_query(inputs) {
                    state.record_attribute(
                        GEN_AI_RETRIEVAL_QUERY_TEXT,
                        redactor.redact(Some(&Value::String(query)), None),
                    );
                }
            }
        });
        if updated.is_none() {
            tracing::debug!(
                target: LOG_TARGET,
                invoke_id = %payload.invoke_id,
                "otel sink: dropping a running workflow frame with no open span"
            );
        }
    }

    /// Closes a workflow span with its terminal status.
    ///
    /// Python counterpart: `on_call_done` (finish), the failure branch of
    /// `on_invoke` (error), and its `GraphInterrupt` branch (interrupt). A
    /// terminal frame that is the first frame of its invoke id opens the span
    /// first — with the frame's start time — so a terminal-only workflow exports
    /// a span with its real duration.
    pub(super) fn finish_workflow(&self, payload: &TraceSpan) {
        let opened_now = if self.workflow_spans.contains(&payload.invoke_id) {
            false
        } else if self.open_workflow_lazily(payload) {
            tracing::debug!(
                target: LOG_TARGET,
                invoke_id = %payload.invoke_id,
                status = payload.status.as_str(),
                "otel sink: no start frame for this workflow span; opening it from the terminal frame"
            );
            true
        } else {
            tracing::debug!(
                target: LOG_TARGET,
                invoke_id = %payload.invoke_id,
                status = payload.status.as_str(),
                "otel sink: dropping a terminal workflow frame with no mappable span"
            );
            return;
        };
        self.feed_buffers(payload);
        // The snapshots must be taken before `pop`: the workflow manager owns
        // the buffers and `with_state`/`pop` hold its lock, so reading a buffer
        // inside a state closure would deadlock.
        let buffered = self.buffered_attributes(&payload.invoke_id);
        let Some(mut state) = self.workflow_spans.pop(&payload.invoke_id) else {
            tracing::debug!(
                target: LOG_TARGET,
                invoke_id = %payload.invoke_id,
                status = payload.status.as_str(),
                "otel sink: dropping a terminal workflow frame with no open span"
            );
            return;
        };
        self.close_workflow_span(&mut state, payload, buffered);
        end_span(&mut state.span, payload, opened_now);
    }

    /// Writes the closing attributes of a workflow span.
    fn close_workflow_span(
        &self,
        state: &mut OtelSpanState,
        payload: &TraceSpan,
        buffered: Vec<(&'static str, String)>,
    ) {
        if !state.recorded {
            return;
        }
        for (key, text) in buffered {
            state.record_attribute(key, text);
        }
        let redactor = Redactor::new(&self.config);
        if let Some(outputs) = present(payload.outputs.as_ref()) {
            state.record_attribute(OJ_WORKFLOW_OUTPUTS, redactor.redact(Some(outputs), None));
            if is_retrieval_state(state) {
                write_retrieval_documents(&mut state.span, outputs, redactor);
            }
        }
        match payload.status {
            SpanStatus::Finish => attrs::apply_end_attributes(&mut state.span, payload),
            SpanStatus::Error => {
                attrs::apply_error_attributes(&mut state.span, payload, AttributeScope::Workflow);
            }
            SpanStatus::Interrupted => {
                attrs::apply_interrupt_attributes(
                    &mut state.span,
                    payload,
                    AttributeScope::Workflow,
                );
            }
            SpanStatus::Start | SpanStatus::Running => {}
        }
    }

    /// Resolves the parent context of a workflow span.
    ///
    /// Python counterpart: `_resolve_parent_context`, which resolves the parent
    /// node through the workflow maps. The workflow manager is consulted first
    /// (a component nests under its host component or root), then the agent
    /// manager (a workflow started inside an agent span).
    fn workflow_parent_context(&self, payload: &TraceSpan) -> Context {
        let parent = payload.parent_invoke_id.as_deref();
        if let Some(span) =
            parent.and_then(|invoke_id| self.workflow_spans.with_state(invoke_id, span_context_of))
        {
            return Context::new().with_remote_span_context(span);
        }
        self.agent_spans.parent_context(parent).unwrap_or_default()
    }

    /// Feeds the three workflow buffers from a frame's `meta_data`.
    ///
    /// This is the frame-model replacement for Python's `on_invoke`
    /// (`on_invoke_data`), `on_pre_stream` (`stream_inputs`), and `on_post_stream`
    /// (`stream_outputs`) callbacks: an array value contributes each element, and
    /// any other value contributes itself. A chunk is dropped when the invoke id
    /// has no open recording span, exactly like the manager's `append_*` methods.
    fn feed_buffers(&self, payload: &TraceSpan) {
        for (key, buffer) in BUFFER_KEYS {
            let Some(value) = payload.meta_data.get(key) else {
                continue;
            };
            match value {
                Value::Array(chunks) => {
                    for chunk in chunks {
                        self.append_buffer(buffer, &payload.invoke_id, chunk.clone());
                    }
                }
                chunk => self.append_buffer(buffer, &payload.invoke_id, chunk.clone()),
            }
        }
    }

    /// Appends one chunk to the buffer of `buffer`.
    fn append_buffer(&self, buffer: Buffer, invoke_id: &str, chunk: Value) {
        match buffer {
            Buffer::InvokeData => self.workflow_spans.append_on_invoke_data(invoke_id, chunk),
            Buffer::StreamInputs => self.workflow_spans.append_stream_input(invoke_id, chunk),
            Buffer::StreamOutputs => self.workflow_spans.append_stream_output(invoke_id, chunk),
        }
    }

    /// Renders the non-empty workflow buffers as span attributes.
    ///
    /// Python counterpart: `_flush_buffered_data`, which writes one compact JSON
    /// attribute per non-empty buffer.
    fn buffered_attributes(&self, invoke_id: &str) -> Vec<(&'static str, String)> {
        let mut attributes = Vec::new();
        for (key, values) in [
            (
                OJ_WORKFLOW_INVOKE_DATA,
                self.workflow_spans.on_invoke_data(invoke_id),
            ),
            (
                OJ_STREAM_INPUTS,
                self.workflow_spans.stream_inputs(invoke_id),
            ),
            (
                OJ_STREAM_OUTPUTS,
                self.workflow_spans.stream_outputs(invoke_id),
            ),
        ] {
            if values.is_empty() {
                continue;
            }
            attributes.push((key, serialize_buffer(&values)));
        }
        attributes
    }
}
