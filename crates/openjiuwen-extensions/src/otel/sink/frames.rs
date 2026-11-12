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

use std::time::{Duration, SystemTime};

use opentelemetry::trace::{Span as _, SpanContext, SpanKind, TraceContextExt as _, Tracer as _};
use opentelemetry::{Array, Context, KeyValue, StringValue, Value as OtelValue};
use opentelemetry_sdk::trace::Span;
use serde_json::{Map, Value};

use openjiuwen_core::session::tracer::{InvokeType, SpanStatus, TraceSpan};

use super::super::attrs::{self, AttributeScope};
use super::super::llm_attrs::{
    extract_retrieval_documents, extract_retrieval_query, redacted_json,
    set_llm_request_attributes, set_llm_response_attributes, standard_messages,
};
use super::super::redaction::{RedactionField, Redactor};
use super::super::semconv::{
    GEN_AI_DATA_SOURCE_ID, GEN_AI_OPERATION_CHAT, GEN_AI_OPERATION_EXECUTE_TOOL,
    GEN_AI_OPERATION_NAME, GEN_AI_OPERATION_RETRIEVAL, GEN_AI_OUTPUT_MESSAGES,
    GEN_AI_RETRIEVAL_DOCUMENTS, GEN_AI_RETRIEVAL_QUERY_TEXT, GEN_AI_RETRIEVAL_TOP_K,
    GEN_AI_TOOL_CALL_ID, GEN_AI_TOOL_NAME, GEN_AI_TOOL_TYPE, OJ_AGENT_INPUTS, OJ_AGENT_INVOKE_TYPE,
    OJ_AGENT_NAME, OJ_AGENT_OUTPUTS, OJ_META_DATA, OJ_SESSION_ID, OJ_STATUS,
    OJ_WORKFLOW_COMPONENT_ID, OJ_WORKFLOW_COMPONENT_NAME, OJ_WORKFLOW_COMPONENT_TYPE,
    OJ_WORKFLOW_LOOP_INDEX, OJ_WORKFLOW_LOOP_NODE_ID, SPAN_STATUS_RUNNING,
};
use super::super::span_manager::OtelSpanState;
use super::super::span_names;
use super::LOG_TARGET;

mod workflow;

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
        if let Some(state) = self.new_agent_state(payload, None) {
            self.agent_spans.push(payload.invoke_id.as_str(), state);
        }
    }

    /// Opens the agent span of a frame that arrives without a start frame.
    ///
    /// The session tracer emits exactly one frame per span — its terminal one —
    /// so the first frame the sink sees for an invoke id is normally `running` or
    /// terminal. Opening the span here keeps the idiomatic
    /// `Tracer::with_sink(..)` + `start_span(..)` + `finish(..)` integration
    /// exporting a span without a synthetic start frame. The frame's own
    /// `start_time` becomes the `OTel` start timestamp so a terminal-only span
    /// still carries a realistic duration.
    ///
    /// Returns `false` when the frame cannot be mapped (no tracer); nothing is
    /// registered in that case.
    fn open_agent_lazily(&self, payload: &TraceSpan) -> bool {
        let Some(state) = self.new_agent_state(payload, frame_time(payload.start_time)) else {
            return false;
        };
        self.agent_spans.push(payload.invoke_id.as_str(), state);
        true
    }

    /// Builds the state of an agent span without registering it.
    ///
    /// `start_time` overrides the span start instant; `None` starts the span
    /// now, which is what a real start frame does.
    fn new_agent_state(
        &self,
        payload: &TraceSpan,
        start_time: Option<SystemTime>,
    ) -> Option<OtelSpanState> {
        let tracer = self.tracer.as_ref()?;
        let invoke_type = payload.invoke_type.unwrap_or_default();
        let entity = entity_name(payload);
        let (name, kind) = agent_span_shape(invoke_type, entity);
        let parent = self.agent_parent_context(payload);
        let builder = tracer.span_builder(name).with_kind(kind);
        let builder = match start_time {
            Some(start_time) => builder.with_start_time(start_time),
            None => builder,
        };
        let span = builder.start_with_context(tracer, &parent);
        let mut state = OtelSpanState::from_span(span, payload.invoke_id.as_str());
        if state.recorded {
            self.write_agent_start(&mut state, payload, invoke_type, entity);
        }
        Some(state)
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
    /// payload rewrites `openjiuwen.agent.inputs`. A running frame that is the
    /// first frame of its invoke id opens the span (and keeps it open, so a later
    /// terminal frame closes it).
    fn running_agent(&self, payload: &TraceSpan) {
        let opened_now = if self.agent_spans.contains(&payload.invoke_id) {
            false
        } else if self.open_agent_lazily(payload) {
            true
        } else {
            tracing::debug!(
                target: LOG_TARGET,
                invoke_id = %payload.invoke_id,
                "otel sink: dropping a running agent frame with no mappable span"
            );
            return;
        };
        let redactor = Redactor::new(&self.config);
        let updated = self.agent_spans.with_state(&payload.invoke_id, |state| {
            state.record_attribute(OJ_STATUS, SPAN_STATUS_RUNNING);
            // A lazily opened span already carries this frame's inputs through
            // its start attributes, so refreshing them would duplicate the write.
            if opened_now {
                return;
            }
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
    /// and the interrupt branch of the agent lifecycle. A terminal frame that is
    /// the first frame of its invoke id opens the span first, writes the start
    /// attributes, and ends it at the frame's end time.
    fn finish_agent(&self, payload: &TraceSpan) {
        let opened_now = if self.agent_spans.contains(&payload.invoke_id) {
            false
        } else if self.open_agent_lazily(payload) {
            tracing::debug!(
                target: LOG_TARGET,
                invoke_id = %payload.invoke_id,
                status = payload.status.as_str(),
                "otel sink: no start frame for this agent span; opening it from the terminal frame"
            );
            true
        } else {
            tracing::debug!(
                target: LOG_TARGET,
                invoke_id = %payload.invoke_id,
                status = payload.status.as_str(),
                "otel sink: dropping a terminal agent frame with no mappable span"
            );
            return;
        };
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
        end_span(&mut state.span, payload, opened_now);
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

    /// Writes the `GenAI` completion messages of an LLM span.
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
/// Python formats `f"retrieval {name}"`; the space-joined `GenAI` style keeps the
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

/// Converts an epoch-millisecond frame timestamp into a [`SystemTime`].
///
/// Returns `None` for an absent timestamp and for a value outside the
/// representable [`SystemTime`] range, so a malformed frame falls back to the
/// current instant instead of panicking.
fn frame_time(millis: Option<u64>) -> Option<SystemTime> {
    millis.and_then(|millis| SystemTime::UNIX_EPOCH.checked_add(Duration::from_millis(millis)))
}

/// Ends a span, honouring the frame's end time only for a lazily opened span.
///
/// A span opened from its terminal frame starts at the frame's `start_time`, so
/// ending it at the frame's `end_time` keeps the exported duration real. A span
/// opened earlier was started "now", which can be later than the frame's end
/// stamp, so it ends now — the same behavior as before lazy opening. Both
/// timestamps are optional: a frame without them falls back to the local clock.
fn end_span(span: &mut Span, payload: &TraceSpan, lazily_opened: bool) {
    if lazily_opened {
        if let (Some(start), Some(end)) =
            (frame_time(payload.start_time), frame_time(payload.end_time))
        {
            if end >= start {
                span.end_with_timestamp(end);
                return;
            }
        }
    }
    span.end();
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
