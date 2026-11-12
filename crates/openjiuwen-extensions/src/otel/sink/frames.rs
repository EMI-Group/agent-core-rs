//! Frame → span mapping for `OtelTraceSink`.
//!
//! Split out of `sink/mod.rs` to keep both files readable: this file owns the
//! agent path (the port of `OtelAgentHandler`) and the workflow path (the port
//! of `OtelWorkflowHandler`), including the component classifier, the
//! `meta_data` → buffer feed, and the buffer flush. Everything here is
//! crate-internal; the public surface lives in `mod.rs`.
//!
//! Every attribute key comes from [`super::super::semconv`]; this module never
//! re-types a wire literal. Attribute *values* are produced by the shared
//! helpers in [`super::super::attrs`] and [`super::super::llm_attrs`], so the
//! frame sink and the typed handlers agree on payload shape.

use opentelemetry::trace::{Span as _, SpanContext, SpanKind, TraceContextExt as _, Tracer as _};
use opentelemetry::{Array, Context, KeyValue, StringValue, Value as OtelValue};
use opentelemetry_sdk::trace::Span;
use serde_json::{Map, Value};

use openjiuwen_core::session::tracer::{InvokeType, SpanStatus, TraceSpan, TraceSpanKind};

use super::super::attrs::{self, AttributeScope};
use super::super::llm_attrs::{
    extract_retrieval_documents, extract_retrieval_query, redacted_json,
    set_llm_request_attributes, set_llm_response_attributes, standard_messages,
};
use super::super::redaction::{RedactionField, Redactor};
use super::super::semconv::{
    GEN_AI_DATA_SOURCE_ID, GEN_AI_OPERATION_CHAT, GEN_AI_OPERATION_EXECUTE_TOOL,
    GEN_AI_OPERATION_NAME, GEN_AI_OPERATION_RETRIEVAL, GEN_AI_OUTPUT_MESSAGES,
    GEN_AI_REQUEST_MODEL, GEN_AI_RETRIEVAL_DOCUMENTS, GEN_AI_RETRIEVAL_QUERY_TEXT,
    GEN_AI_RETRIEVAL_TOP_K, GEN_AI_TOOL_CALL_ID, GEN_AI_TOOL_NAME, GEN_AI_TOOL_TYPE,
    OJ_AGENT_INPUTS, OJ_AGENT_INVOKE_TYPE, OJ_AGENT_NAME, OJ_AGENT_OUTPUTS, OJ_META_DATA,
    OJ_PARENT_NODE_ID, OJ_SESSION_ID, OJ_SOURCE_IDS, OJ_STATUS, OJ_STREAM_INPUTS,
    OJ_STREAM_OUTPUTS, OJ_WORKFLOW_COMPONENT_ID, OJ_WORKFLOW_COMPONENT_NAME,
    OJ_WORKFLOW_COMPONENT_TYPE, OJ_WORKFLOW_EXECUTION_ID, OJ_WORKFLOW_ID, OJ_WORKFLOW_INPUTS,
    OJ_WORKFLOW_INVOKE_DATA, OJ_WORKFLOW_LOOP_INDEX, OJ_WORKFLOW_LOOP_NODE_ID, OJ_WORKFLOW_NAME,
    OJ_WORKFLOW_OUTPUTS, OJ_WORKFLOW_VERSION, SPAN_STATUS_RUNNING,
};
use super::super::span_manager::OtelSpanState;
use super::super::span_names;
use super::LOG_TARGET;

/// Role assumed for an LLM output message that does not declare one.
///
/// Python counterpart: the `default_role="assistant"` argument of
/// `_standard_messages` in `on_llm_end`.
const DEFAULT_OUTPUT_ROLE: &str = "assistant";

/// Container keys an LLM output payload may wrap its messages in.
///
/// Python counterpart: `_unwrap_llm_payload`, applied by `on_llm_end` before
/// `_standard_messages`; the first present key wins.
const OUTPUT_CONTAINER_KEYS: [&str; 3] = ["outputs", "messages", "inputs"];

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

impl super::OtelTraceSink {
    /// Maps one agent-dimension frame.
    pub(super) fn emit_agent(&self, payload: &TraceSpan) {
        match payload.status {
            SpanStatus::Start => self.start_agent(payload),
            SpanStatus::Running => self.running_agent(payload),
            SpanStatus::Finish | SpanStatus::Error | SpanStatus::Interrupted => {
                self.finish_agent(payload);
            }
        }
    }

    /// Maps one workflow/component frame.
    pub(super) fn emit_workflow(&self, payload: &TraceSpan) {
        match payload.status {
            SpanStatus::Start => self.start_workflow_frame(payload),
            SpanStatus::Running => self.running_workflow(payload),
            SpanStatus::Finish | SpanStatus::Error | SpanStatus::Interrupted => {
                self.finish_workflow(payload);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Agent path — port of `OtelAgentHandler`
    // -----------------------------------------------------------------------

    /// Starts an agent span.
    ///
    /// Python counterpart: `_start_and_push` plus the invoke-type branches of
    /// `on_*_start`. A non-recording span is still registered, so its terminal
    /// frame pops a state instead of logging an unknown invoke id.
    fn start_agent(&self, payload: &TraceSpan) {
        let Some(tracer) = self.tracer.as_ref() else {
            return;
        };
        let invoke_type = payload.invoke_type.unwrap_or_default();
        let entity = entity_name(payload);
        let (name, kind) = agent_span_shape(invoke_type, entity);
        let parent = self.agent_parent_context(payload);
        let span = tracer
            .span_builder(name)
            .with_kind(kind)
            .start_with_context(tracer, &parent);
        let mut state = OtelSpanState::from_span(span, payload.invoke_id.as_str());
        if state.recorded {
            self.write_agent_start(&mut state, payload, invoke_type, entity);
        }
        self.agent_spans.push(payload.invoke_id.as_str(), state);
    }

    /// Applies the sink's fallback session id when the frame carries none.
    ///
    /// Python counterpart: the tracer-injected `_session_id` fallback both
    /// handlers keep; a session frame normally carries its own id, so this only
    /// fills the gap for a frame built without one.
    fn write_session_fallback(&self, state: &mut OtelSpanState, payload: &TraceSpan) {
        if payload.session_id.is_empty() {
            if let Some(session_id) = self.fallback_session_id() {
                state.record_attribute(OJ_SESSION_ID, session_id);
            }
        }
    }

    /// Writes the start attributes of a recording agent span.
    ///
    /// Python counterpart: the base fields of `_start_and_push` plus the
    /// `openjiuwen.agent.*` block every agent callback writes.
    fn write_agent_start(
        &self,
        state: &mut OtelSpanState,
        payload: &TraceSpan,
        invoke_type: InvokeType,
        entity: &str,
    ) {
        attrs::apply_start_attributes(&mut state.span, payload);
        self.write_session_fallback(state, payload);
        state.record_attribute(OJ_AGENT_INVOKE_TYPE, invoke_type.as_str());
        state.record_attribute(OJ_AGENT_NAME, entity);
        if !payload.meta_data.is_empty() {
            state.record_attribute(
                OJ_META_DATA,
                attrs::serialize_value(&metadata_value(payload)),
            );
        }
        let redactor = Redactor::new(&self.config);
        let inputs = present(payload.inputs.as_ref());
        if let Some(inputs) = inputs {
            state.record_attribute(
                OJ_AGENT_INPUTS,
                redactor.redact(Some(inputs), Some(RedactionField::Prompts)),
            );
        }
        self.write_invoke_type_attributes(state, payload, invoke_type, entity, inputs, redactor);
    }

    /// Writes the invoke-type specific attributes of an agent span.
    ///
    /// Python counterpart: the `gen_ai.*` blocks of `on_llm_start`,
    /// `on_plugin_start`, and `on_retriever_start`; prompt/chain/evaluator/
    /// workflow spans carry no extra attributes beyond the common ones.
    fn write_invoke_type_attributes(
        &self,
        state: &mut OtelSpanState,
        payload: &TraceSpan,
        invoke_type: InvokeType,
        entity: &str,
        inputs: Option<&Value>,
        redactor: Redactor<'_>,
    ) {
        match invoke_type {
            InvokeType::Llm => {
                state.record_attribute(GEN_AI_OPERATION_NAME, GEN_AI_OPERATION_CHAT);
                if let Some(inputs) = inputs {
                    let info = instance_info(payload);
                    set_llm_request_attributes(&mut state.span, inputs, &info, &self.config);
                }
            }
            InvokeType::Plugin => {
                state.record_attribute(GEN_AI_OPERATION_NAME, GEN_AI_OPERATION_EXECUTE_TOOL);
                state.record_attribute(GEN_AI_TOOL_NAME, entity);
                for (source, key) in [
                    ("tool_call_id", GEN_AI_TOOL_CALL_ID),
                    ("tool_type", GEN_AI_TOOL_TYPE),
                ] {
                    if let Some(text) = truthy_text(payload.meta_data.get(source)) {
                        state.record_attribute(key, text);
                    }
                }
            }
            InvokeType::Retriever => {
                state.record_attribute(GEN_AI_OPERATION_NAME, GEN_AI_OPERATION_RETRIEVAL);
                let Some(inputs) = inputs else {
                    return;
                };
                if let Some(object) = inputs.as_object() {
                    if let Some(top_k) = pick(object, &["top_k", "k"]).and_then(json_i64) {
                        state
                            .span
                            .set_attribute(KeyValue::new(GEN_AI_RETRIEVAL_TOP_K, top_k));
                    }
                    if let Some(source) = pick(object, &["data_source_id", "data_source"]) {
                        state.record_attribute(
                            GEN_AI_DATA_SOURCE_ID,
                            attrs::serialize_value(source),
                        );
                    }
                }
                if let Some(query) = extract_retrieval_query(inputs) {
                    state.record_attribute(
                        GEN_AI_RETRIEVAL_QUERY_TEXT,
                        redactor.redact(Some(&Value::String(query)), None),
                    );
                }
            }
            InvokeType::Prompt
            | InvokeType::Chain
            | InvokeType::Evaluator
            | InvokeType::Workflow => {}
        }
    }

    /// Updates the open span of a running agent invocation.
    ///
    /// Python counterpart: `on_llm_request` (the only mid-flight agent event the
    /// frame model keeps). The status flips to `running` and a fresh inputs
    /// payload rewrites `openjiuwen.agent.inputs`.
    fn running_agent(&self, payload: &TraceSpan) {
        let redactor = Redactor::new(&self.config);
        let updated = self.agent_spans.with_state(&payload.invoke_id, |state| {
            state.record_attribute(OJ_STATUS, SPAN_STATUS_RUNNING);
            if let Some(inputs) = present(payload.inputs.as_ref()) {
                state.record_attribute(
                    OJ_AGENT_INPUTS,
                    redactor.redact(Some(inputs), Some(RedactionField::Prompts)),
                );
            }
        });
        if updated.is_none() {
            tracing::debug!(
                target: LOG_TARGET,
                invoke_id = %payload.invoke_id,
                "otel sink: dropping a running agent frame with no open span"
            );
        }
    }

    /// Closes an agent span with its terminal status.
    ///
    /// Python counterpart: `on_*_end` plus `_end_and_pop`, `_mark_error_and_end`,
    /// and the interrupt branch of the agent lifecycle.
    fn finish_agent(&self, payload: &TraceSpan) {
        let Some(mut state) = self.agent_spans.pop(&payload.invoke_id) else {
            tracing::debug!(
                target: LOG_TARGET,
                invoke_id = %payload.invoke_id,
                status = payload.status.as_str(),
                "otel sink: dropping a terminal agent frame with no open span"
            );
            return;
        };
        self.close_agent_span(&mut state, payload);
        state.span.end();
    }

    /// Writes the closing attributes of an agent span.
    fn close_agent_span(&self, state: &mut OtelSpanState, payload: &TraceSpan) {
        if !state.recorded {
            return;
        }
        match payload.status {
            SpanStatus::Finish => {
                attrs::apply_end_attributes(&mut state.span, payload);
                let redactor = Redactor::new(&self.config);
                let outputs = present(payload.outputs.as_ref());
                if let Some(outputs) = outputs {
                    state.record_attribute(
                        OJ_AGENT_OUTPUTS,
                        redactor.redact(Some(outputs), Some(RedactionField::Completions)),
                    );
                }
                match payload.invoke_type.unwrap_or_default() {
                    InvokeType::Llm => {
                        if let Some(outputs) = outputs {
                            Self::write_output_messages(state, outputs, redactor);
                            set_llm_response_attributes(&mut state.span, outputs);
                        }
                    }
                    InvokeType::Retriever => {
                        if let Some(outputs) = outputs {
                            write_retrieval_documents(&mut state.span, outputs, redactor);
                        }
                    }
                    _ => {}
                }
            }
            SpanStatus::Error => {
                attrs::apply_error_attributes(&mut state.span, payload, AttributeScope::Agent);
            }
            SpanStatus::Interrupted => {
                attrs::apply_interrupt_attributes(&mut state.span, payload, AttributeScope::Agent);
            }
            SpanStatus::Start | SpanStatus::Running => {}
        }
    }

    /// Writes the GenAI completion messages of an LLM span.
    ///
    /// Python counterpart: the `outputs is not None` block of `on_llm_end` —
    /// unwrap the container, normalize with the `assistant` default role, and
    /// redact the result under the completion policy.
    fn write_output_messages(state: &mut OtelSpanState, outputs: &Value, redactor: Redactor<'_>) {
        let (_, messages) = standard_messages(output_payload(outputs), DEFAULT_OUTPUT_ROLE);
        if messages.is_empty() {
            return;
        }
        let text = redacted_json(
            &Value::Array(messages),
            &redactor,
            RedactionField::Completions,
            false,
        );
        attrs::set_text_attribute(&mut state.span, GEN_AI_OUTPUT_MESSAGES, &text);
    }

    /// Resolves the parent context of an agent span.
    ///
    /// Python counterpart: `_resolve_parent_context`, which only consults the
    /// agent span manager. A frame model may parent an agent span onto a
    /// workflow component (an LLM call inside a workflow node), so the workflow
    /// manager is the documented second source; a missing parent makes the span
    /// a trace root.
    fn agent_parent_context(&self, payload: &TraceSpan) -> Context {
        let parent = payload.parent_invoke_id.as_deref();
        if let Some(context) = self.agent_spans.parent_context(parent) {
            return context;
        }
        parent
            .and_then(|invoke_id| self.workflow_spans.with_state(invoke_id, span_context_of))
            .map_or_else(Context::new, |span| {
                Context::new().with_remote_span_context(span)
            })
    }

    // -----------------------------------------------------------------------
    // Workflow path — port of `OtelWorkflowHandler`
    // -----------------------------------------------------------------------

    /// Starts a workflow or component span.
    ///
    /// Python counterpart: `on_call_start`. The span is registered before the
    /// buffers are fed, because appending requires an open state.
    fn start_workflow_frame(&self, payload: &TraceSpan) {
        let Some(tracer) = self.tracer.as_ref() else {
            return;
        };
        let family = WorkflowFamily::classify(payload);
        let name = family.span_name(payload);
        let parent = self.workflow_parent_context(payload);
        let span = tracer
            .span_builder(name)
            .with_kind(family.span_kind())
            .start_with_context(tracer, &parent);
        let mut state = OtelSpanState::from_span(span, payload.invoke_id.as_str());
        state.workflow_id.clone_from(&payload.workflow_id);
        if state.recorded {
            self.write_workflow_start(&mut state, payload, family);
        }
        self.workflow_spans.push(payload.invoke_id.as_str(), state);
        self.feed_buffers(payload);
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
    /// buffered chunks accumulate for the terminal flush.
    fn running_workflow(&self, payload: &TraceSpan) {
        self.feed_buffers(payload);
        let redactor = Redactor::new(&self.config);
        let updated = self.workflow_spans.with_state(&payload.invoke_id, |state| {
            state.record_attribute(OJ_STATUS, SPAN_STATUS_RUNNING);
            if let Some(inputs) = present(payload.inputs.as_ref()) {
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
    /// `on_invoke` (error), and its `GraphInterrupt` branch (interrupt).
    fn finish_workflow(&self, payload: &TraceSpan) {
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
        state.span.end();
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

/// The agent span name and kind of an invoke type.
///
/// Python counterpart: `on_llm_start` (`chat {class_name}`, [`SpanKind::Client`]),
/// `on_plugin_start` (`execute_tool {name}`), and the `{prefix}.{name}` names of
/// `_start_non_llm_span`, all [`SpanKind::Internal`].
fn agent_span_shape(invoke_type: InvokeType, entity: &str) -> (String, SpanKind) {
    match invoke_type {
        InvokeType::Llm if entity.is_empty() => {
            (span_names::chat_span_name_for_model(""), SpanKind::Client)
        }
        InvokeType::Llm => (span_names::chat_span_name(entity), SpanKind::Client),
        InvokeType::Plugin => (span_names::tool_span_name(entity), SpanKind::Internal),
        other => (
            span_names::simple_span_name(other, entity),
            SpanKind::Internal,
        ),
    }
}

/// The name an agent frame's span is built from: its name, else its invoke id.
fn entity_name(payload: &TraceSpan) -> &str {
    if payload.name.is_empty() {
        &payload.invoke_id
    } else {
        &payload.name
    }
}

/// The entity name of a workflow component: its component name, else its invoke id.
///
/// Python counterpart: `str(metadata.get("component_name") or invoke_id)`.
fn component_entity(payload: &TraceSpan) -> &str {
    payload
        .component_name
        .as_deref()
        .filter(|name| !name.is_empty())
        .unwrap_or(&payload.invoke_id)
}

/// The requested model of an LLM component: `model_name`, else `model`.
fn model_name(payload: &TraceSpan) -> Option<String> {
    truthy_text(payload.meta_data.get("model_name"))
        .or_else(|| truthy_text(payload.meta_data.get("model")))
}

/// The span name of a retrieval component: `retrieval {name}`.
///
/// Python formats `f"retrieval {name}"`; the space-joined GenAI style keeps the
/// bare operation name when the component name is blank, the same rule as
/// [`span_names::tool_span_name`].
fn retrieval_span_name(name: &str) -> String {
    if name.trim().is_empty() {
        return GEN_AI_OPERATION_RETRIEVAL.to_owned();
    }
    format!("{GEN_AI_OPERATION_RETRIEVAL} {name}")
}

/// Writes the component identity attributes of a workflow span.
///
/// Python counterpart: the component branch of `_set_workflow_attrs` — the
/// component id/type/name plus the loop coordinates when the node runs inside a
/// loop.
fn write_component_identity(state: &mut OtelSpanState, payload: &TraceSpan) {
    state.record_attribute(
        OJ_WORKFLOW_COMPONENT_ID,
        payload.component_id.clone().unwrap_or_default(),
    );
    state.record_attribute(
        OJ_WORKFLOW_COMPONENT_TYPE,
        payload.component_type.clone().unwrap_or_default(),
    );
    state.record_attribute(
        OJ_WORKFLOW_COMPONENT_NAME,
        payload.component_name.clone().unwrap_or_default(),
    );
    for (source, key) in [
        ("loop_node_id", OJ_WORKFLOW_LOOP_NODE_ID),
        ("loop_index", OJ_WORKFLOW_LOOP_INDEX),
    ] {
        if let Some(value) = present(payload.meta_data.get(source)) {
            state.record_attribute(key, attrs::serialize_value(value));
        }
    }
}

/// Renders a metadata field as attribute text, defaulting to the empty string.
///
/// Python counterpart: `metadata.get(key, "")`; a non-string value is rendered
/// as text so the attribute stays a string.
fn meta_text(payload: &TraceSpan, key: &str) -> String {
    payload
        .meta_data
        .get(key)
        .map_or_else(String::new, attrs::serialize_value)
}

/// The `meta_data` map of a frame as a JSON object.
fn metadata_value(payload: &TraceSpan) -> Value {
    Value::Object(
        payload
            .meta_data
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
    )
}

/// The rail-forwarded instance info of an LLM frame.
///
/// Python counterpart: the `instance_info` dict `OtelRail` forwards. The frame
/// model has no separate channel for it, so the span metadata doubles as the
/// info map, with the span's own name as the `class_name`/`agent_name` default
/// (a present key is never overwritten).
fn instance_info(payload: &TraceSpan) -> Value {
    let mut info: Map<String, Value> = payload
        .meta_data
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    if !payload.name.is_empty() {
        for key in ["class_name", "agent_name"] {
            if !info.contains_key(key) {
                info.insert(key.to_owned(), Value::String(payload.name.clone()));
            }
        }
    }
    Value::Object(info)
}

/// The value an LLM output payload wraps its messages in.
///
/// Python counterpart: `_unwrap_llm_payload` as called by `on_llm_end`; the
/// first present container key wins, and any other shape passes through.
fn output_payload(outputs: &Value) -> &Value {
    outputs
        .as_object()
        .and_then(|object| {
            OUTPUT_CONTAINER_KEYS
                .iter()
                .find_map(|key| object.get(*key))
        })
        .unwrap_or(outputs)
}

/// Writes the redacted retrieval documents of `outputs` onto a span.
///
/// Python counterpart: the `_is_retrieval_component_span` /
/// `_extract_retrieval_documents` block of `on_call_done` and `on_retriever_end`.
/// Documents follow the same policy as the query text and the span outputs:
/// hashed under redaction, truncated otherwise — never raw document text.
fn write_retrieval_documents(span: &mut Span, outputs: &Value, redactor: Redactor<'_>) {
    let Some(documents) = extract_retrieval_documents(outputs) else {
        return;
    };
    let documents = documents
        .iter()
        .map(|document| redactor.redact(Some(&Value::String(document.clone())), None))
        .collect::<Vec<_>>();
    span.set_attribute(KeyValue::new(
        GEN_AI_RETRIEVAL_DOCUMENTS,
        string_array(documents),
    ));
}

/// Whether an open state belongs to a retrieval component.
///
/// Python counterpart: `_is_retrieval_component_span`, which reads the span's
/// `openjiuwen.workflow.component.type`. The SDK span does not expose its
/// attributes, so the value comes from [`OtelSpanState`]'s cache — which is why
/// the start path records the component type through
/// [`OtelSpanState::record_attribute`].
fn is_retrieval_state(state: &OtelSpanState) -> bool {
    state
        .attribute(OJ_WORKFLOW_COMPONENT_TYPE)
        .is_some_and(span_names::is_retrieval_component)
}

/// Extracts the span identity of an open state.
///
/// A free function (not a closure) so it can be handed to
/// [`super::super::span_manager::OtelWorkflowSpanManager::with_state`], whose
/// closure receives `&mut OtelSpanState`.
fn span_context_of(state: &mut OtelSpanState) -> SpanContext {
    state.span_context()
}

/// Wraps strings in an `OTel` array attribute value.
fn string_array(values: impl IntoIterator<Item = String>) -> OtelValue {
    OtelValue::Array(Array::String(
        values.into_iter().map(StringValue::from).collect(),
    ))
}

/// Serializes a buffered payload list as compact JSON.
///
/// Python counterpart: `_serialize`, applied by `_flush_buffered_data` to a
/// whole buffer. Serializing a [`Value`] cannot fail (no non-string map keys,
/// no non-finite numbers), so the empty fallback is unreachable and keeps
/// [`super::TraceSink::emit`] panic-free.
fn serialize_buffer(values: &[Value]) -> String {
    serde_json::to_string(values).unwrap_or_else(|_| "[]".to_owned())
}

/// Filters out an absent or `null` payload.
fn present(value: Option<&Value>) -> Option<&Value> {
    value.filter(|value| !value.is_null())
}

/// Reads a JSON value as an integer, mirroring Python's `int()` coercion.
///
/// A JSON integer converts directly and a float truncates toward zero; a
/// numeric string parses after trimming. Any other shape yields `None`.
#[allow(clippy::cast_possible_truncation)]
fn json_i64(value: &Value) -> Option<i64> {
    match value {
        Value::Number(number) => number
            .as_i64()
            .or_else(|| number.as_f64().map(|float| float as i64)),
        Value::String(text) => text.trim().parse().ok(),
        _ => None,
    }
}

/// Returns the first truthy value among `keys`, mirroring Python's `a or b`.
fn pick<'a>(object: &'a Map<String, Value>, keys: &[&str]) -> Option<&'a Value> {
    keys.iter()
        .find_map(|key| object.get(*key).filter(|value| is_truthy(value)))
}

/// Reports whether a JSON value is truthy in Python's `if value:` sense.
///
/// `null`, `false`, `0`, the empty string, and empty containers are falsy;
/// everything else is truthy. The retrieval and tool blocks use bare truthiness
/// checks (`if data_source is not None`, `if tool_call_id:`).
fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|number| number != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}

/// Renders a metadata field the way Python's `str(value) if value else ""` does.
///
/// Strings are used verbatim and an empty string counts as absent; every other
/// non-null JSON value is rendered as text, so `12` becomes `"12"`.
fn truthy_text(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::Null => None,
        Value::String(text) if text.is_empty() => None,
        Value::String(text) => Some(text.clone()),
        other => Some(attrs::serialize_value(other)),
    }
}
