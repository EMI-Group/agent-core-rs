//! Agent-dimension `OTel` handler: agent trace frames become `OTel` spans.
//!
//! Port of `OtelAgentHandler` from
//! `openjiuwen/extensions/tracer_otel/handler.py` (v0.1.20). The handler maps
//! the session tracer's agent frames onto `OTel` spans with Python's names,
//! kinds, attributes, statuses, and parents:
//!
//! - an LLM invocation is a `chat {class_name}` span of [`SpanKind::Client`]
//!   carrying the `gen_ai.*` request/response conventions;
//! - a tool invocation is an `execute_tool {class_name}` span of
//!   [`SpanKind::Internal`] carrying the GenAI tool identity;
//! - every other invocation is `{prefix}.{class_name}` (prompt, chain,
//!   retriever, memory, evaluator, workflow) or `embeddings {model}`, all
//!   [`SpanKind::Internal`] with the `openjiuwen.agent.*` attributes.
//!
//! Open spans live in an [`OtelAgentSpanManager`] keyed by invoke id, so a child
//! frame whose `parent_invoke_id` names an open span continues that trace; the
//! same map drives the tool-span dedup of `bug #2046`.
//!
//! # Layout
//!
//! [`mod.rs`](self) owns the handler state, the lifecycle helpers shared by both
//! span families, and the LLM callbacks; [`non_llm`] owns the tool, prompt,
//! chain, retriever, embedding, memory, evaluator, and workflow callbacks.
//! [`tests`] holds the unit tests.
//!
//! # Deviations from Python
//!
//! - Python's callbacks take keyword arguments (`**kwargs`) that Rust trace
//!   frames do not carry; the signatures take exactly the facts each callback
//!   consumes, and
//!   [`OtelAgentHandler::on_llm_request`] emits a *bare* `llm.request` event
//!   instead of forwarding the request keyword arguments.
//! - No ambient-context attach/detach: Rust has no `otel_context.attach` token
//!   on the session path, so [`OtelSpanState`] stores no token and
//!   [`OtelAgentSpanManager::parent_context`] rebuilds the parent context from
//!   the open parent span's identity.
//! - The SDK span does not expose its attributes, so the identity the dedup
//!   re-reads (`openjiuwen.trace.id`, `gen_ai.tool.name`,
//!   `gen_ai.operation.name`) is written through
//!   [`OtelSpanState::record_attribute`], which caches it.
//! - [`OtelAgentHandler::start_and_push`] forces `recorded = true`, exactly like
//!   Python's `_start_and_push`; the `if not state.recorded: return` guards of
//!   the LLM callbacks are kept for parity with Python's control flow.
//! - `openjiuwen.agent.inputs` is redacted with
//!   [`RedactionField::Prompts`], while Python's `_start_non_llm_span` calls
//!   `redact(inputs, config)` without a field (i.e. it follows
//!   `redaction_enabled`); the objective of this port pins the prompts field, so
//!   a `redact_prompts` override applies to non-LLM inputs as well.
//! - The tool-dedup probe scopes by the handler's trace id (Python reads the id
//!   off the frame) and falls back to the frame's `name` when
//!   `instance_info["class_name"]` is empty.
//! - Failure attributes use [`openjiuwen_common::error::ErrorKind::as_str`] as
//!   the `openjiuwen.error` code, where Python uses `BaseError.status.code` and
//!   falls back to `100102` for foreign exceptions.
//! - [`attrs::apply_end_attributes`] also sets the `OK` status, which Python
//!   splits between `_set_end_attrs` and `_end_and_pop`.
//! - Fail-open: every callback is infallible by construction (no `unwrap` on
//!   payload data, no panics) and anomalies are logged at `warn` instead of
//!   being returned, matching Python's blanket `try/except`.
//!
//! [`SpanKind::Client`]: opentelemetry::trace::SpanKind::Client
//! [`SpanKind::Internal`]: opentelemetry::trace::SpanKind::Internal

// The otel `mod.rs` re-exports of this module land in a follow-up commit owned by
// the parent agent, so nothing in the crate calls the handler yet and its `pub`
// items are not reachable from outside the module tree. The `unused_async`
// opt-out is deliberate as well: the callbacks mirror Python's coroutines and the
// session-callback contract, while their bodies stay fully synchronous so no lock
// is ever held across an `await`. All of these go away with the re-export.
#![allow(
    dead_code,
    unreachable_pub,
    clippy::unused_async,
    clippy::unused_async_trait_impl
)]

mod non_llm;
#[cfg(test)]
mod tests;

use opentelemetry::trace::{Span as _, SpanKind, Status, Tracer as _};
use opentelemetry::Context;
use opentelemetry_sdk::trace::SdkTracer;
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_core::session::tracer::{InvokeType, TraceSpan};

use super::attrs::{self, AttributeScope};
use super::config::OtelTracerConfig;
use super::handle::TracerHandle;
use super::llm_attrs;
use super::redaction::{RedactionField, Redactor};
use super::semconv::{
    GEN_AI_AGENT_DESCRIPTION, GEN_AI_AGENT_ID, GEN_AI_AGENT_NAME, GEN_AI_CONVERSATION_ID,
    GEN_AI_OPERATION_EXECUTE_TOOL, GEN_AI_OPERATION_NAME, GEN_AI_OUTPUT_MESSAGES, OJ_AGENT_INPUTS,
    OJ_AGENT_INVOKE_TYPE, OJ_AGENT_NAME, OJ_AGENT_OUTPUTS, OJ_GEN_AI_METADATA,
    OJ_GEN_AI_TRACE_NAME, OJ_GEN_AI_USER_ID, OJ_SESSION_ID, OJ_TRACE_ID,
};
use super::span_manager::{OtelAgentSpanManager, OtelSpanState};
use super::span_names::chat_span_name;

/// `tracing` target of every warning this module family emits.
const LOG_TARGET: &str = "openjiuwen_extensions::otel::agent_handler";

/// `instance_info["type"]` of the tool rail's own plugin event.
///
/// Python counterpart: `InvokeType.PLUGIN.value` (`"plugin"`), which
/// `on_plugin_start` compares against to recognise the rail's event.
const RAIL_PLUGIN_EVENT: &str = "plugin";

/// `instance_info["type"]` of the legacy session-tracer plugin event.
///
/// Python counterpart: the `type="tool"` fingerprint of
/// `_is_duplicate_tool_event`; only this shape is the decorator twin of an open
/// rail span (`bug #2046`).
const DECORATOR_PLUGIN_EVENT: &str = "tool";

/// Default `class_name` the LLM span name uses when the rail forwards none.
///
/// Python counterpart: `instance_info.get("class_name", "unknown")`.
const UNKNOWN_CLASS_NAME: &str = "unknown";

/// Start parameters of a non-LLM span.
///
/// Python counterpart: the positional `invoke_type`, `span_name_prefix`,
/// `extra_attrs`, and `span_name` arguments of `_start_non_llm_span`, bundled so
/// the helper keeps a readable arity.
struct NonLlmSpan {
    /// `openjiuwen.agent.invoke_type` value used when the frame has none.
    invoke_type: &'static str,
    /// Prefix of the `{prefix}.{class_name}` span name.
    prefix: &'static str,
    /// Explicit span name overriding the prefix rule (the embeddings rule).
    span_name: Option<String>,
    /// Event-specific extra attributes (`gen_ai.*` identity of a tool/embedding).
    extra_attrs: Vec<(&'static str, String)>,
}

impl NonLlmSpan {
    /// Builds the specification of a `{prefix}.{class_name}` span.
    fn new(invoke_type: &'static str, prefix: &'static str) -> Self {
        Self {
            invoke_type,
            prefix,
            span_name: None,
            extra_attrs: Vec::new(),
        }
    }
}

/// Agent-dimension handler: turns agent frames into `OTel` spans.
///
/// Build one per tracer session with [`Self::new`] or from an initialized
/// [`TracerHandle`] with [`Self::from_handle`], then feed it the agent
/// callbacks of the session tracer (`on_llm_start`, `on_plugin_end`, …). Every
/// callback is fail-open: failures are logged at `warn` and never reach the
/// caller.
///
/// The handler is `Send + Sync`, so one instance can serve frames from several
/// tasks; the open spans sit behind a mutex that is never held across an
/// `await`.
#[derive(Debug)]
pub struct OtelAgentHandler {
    /// Tracer bound to the private provider.
    tracer: SdkTracer,
    /// Configuration driving redaction and field overrides.
    config: OtelTracerConfig,
    /// Trace id of the run, used to scope the tool-span dedup.
    trace_id: String,
    /// Session id used when a frame carries none.
    session_id: Option<String>,
    /// Open spans, keyed by invoke id.
    span_manager: OtelAgentSpanManager,
}

impl OtelAgentHandler {
    /// Builds a handler over `tracer`.
    ///
    /// `trace_id` is the id used to scope the tool-span dedup, like Python's
    /// optional `trace_id` constructor argument; `None` starts with an empty id,
    /// which matches any trace in
    /// [`OtelAgentSpanManager::has_open_tool_span`].
    #[must_use]
    pub fn new(tracer: SdkTracer, config: OtelTracerConfig, trace_id: Option<String>) -> Self {
        Self {
            tracer,
            config,
            trace_id: trace_id.unwrap_or_default(),
            session_id: None,
            span_manager: OtelAgentSpanManager::new(),
        }
    }

    /// Builds a handler from an initialized [`TracerHandle`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::config`] when the handle is inactive (built without the
    /// `otel` feature, or already shut down), because no tracer is available.
    pub fn from_handle(handle: &TracerHandle) -> Result<Self> {
        let tracer = handle
            .tracer()
            .filter(|_| handle.is_active())
            .ok_or_else(|| {
                Error::config(
                    "the otel tracer handle is inactive; initialize one with init_otel_tracer",
                )
            })?;
        Ok(Self::new(tracer, handle.config().clone(), None))
    }

    /// Sets the trace id used to scope the tool-span dedup.
    ///
    /// Python counterpart: `set_trace_id`. Open spans are kept: Python assigns
    /// the new id without touching the span manager either.
    pub fn set_trace_id(&mut self, trace_id: impl Into<String>) {
        self.trace_id = trace_id.into();
    }

    /// Sets the session id written when a frame carries none.
    ///
    /// Python counterpart: `set_session_id`. A frame's own `session_id` always
    /// wins, so a globally registered handler can serve several sessions.
    pub fn set_session_id(&mut self, session_id: impl Into<String>) {
        self.session_id = Some(session_id.into());
    }

    /// Reports whether an open tool span of the tool rail matches `tool_name`.
    ///
    /// Delegates to [`OtelAgentSpanManager::has_open_tool_span`], the probe that
    /// drops the session tracer's duplicate tool event (`bug #2046`). An empty
    /// `trace_id` matches any trace.
    #[must_use]
    pub fn has_open_tool_span(&self, tool_name: &str, trace_id: &str) -> bool {
        self.span_manager.has_open_tool_span(tool_name, trace_id)
    }

    /// Starts the span of an LLM invocation.
    ///
    /// Python counterpart: `on_llm_start` — a `chat {class_name}`
    /// [`SpanKind::Client`] span with the base fields, the `gen_ai.*` request
    /// attributes (see [`llm_attrs::set_llm_request_attributes`]), and the
    /// `openjiuwen.agent.*` identity pair. A non-recording state skips the
    /// attribute block exactly like Python's early return.
    pub async fn on_llm_start(&self, span: &TraceSpan, inputs: &Value, instance_info: &Value) {
        let class_name = text_field(instance_info, "class_name");
        let span_name = chat_span_name(class_name.unwrap_or(UNKNOWN_CLASS_NAME));
        self.start_and_push(&span_name, SpanKind::Client, span);
        // Python falls back to `instance_info.get("class_name", "")`, an empty
        // string when the rail forwarded none.
        let fallback_name = class_name.unwrap_or_default();
        self.span_manager.with_state(&span.invoke_id, |state| {
            if !state.recorded {
                return;
            }
            llm_attrs::set_llm_request_attributes(
                &mut state.span,
                inputs,
                instance_info,
                &self.config,
            );
            let invoke_type = span.invoke_type.unwrap_or(InvokeType::Llm);
            state.record_attribute(OJ_AGENT_INVOKE_TYPE, invoke_type.as_str());
            let name = if span.name.is_empty() {
                fallback_name
            } else {
                span.name.as_str()
            };
            state.record_attribute(OJ_AGENT_NAME, name);
        });
    }

    /// Records that the LLM request is being sent.
    ///
    /// Python counterpart: `on_llm_request`, which attaches the callback's
    /// keyword arguments to the `llm.request` event. Rust frames carry no such
    /// keyword arguments, so the event is emitted bare (see the module docs).
    pub async fn on_llm_request(&self, span: &TraceSpan) {
        self.span_manager.with_state(&span.invoke_id, |state| {
            if !state.recorded {
                return;
            }
            state.span.add_event("llm.request", Vec::new());
        });
    }

    /// Closes the span of a successful LLM invocation.
    ///
    /// Python counterpart: `on_llm_end` — the redacted completion messages
    /// (`gen_ai.output.messages`), the response/usage/cost attributes, the
    /// end-time fields, then an `OK` span. A frame with no open span is ignored;
    /// a non-recording span is ended without any attribute.
    pub async fn on_llm_end(&self, span: &TraceSpan, outputs: &Value) {
        if !self.span_manager.contains(&span.invoke_id) {
            return;
        }
        let redactor = Redactor::new(&self.config);
        self.span_manager.with_state(&span.invoke_id, |state| {
            if !state.recorded {
                return;
            }
            if !outputs.is_null() {
                let response = llm_attrs::unwrap_llm_payload(outputs);
                let (_, messages) = llm_attrs::standard_messages(&response, "assistant");
                if !messages.is_empty() {
                    let payload = llm_attrs::redacted_json(
                        &serde_json::json!(messages),
                        &redactor,
                        RedactionField::Completions,
                        false,
                    );
                    state.record_attribute(GEN_AI_OUTPUT_MESSAGES, payload);
                }
                llm_attrs::set_llm_response_attributes(&mut state.span, outputs);
            }
            set_end_attrs(state, span);
        });
        self.end_and_pop(&span.invoke_id);
    }

    /// Closes the span of a failed LLM invocation.
    ///
    /// Python counterpart: `on_llm_error` → `_mark_error_and_end`. A frame with
    /// no open span is ignored, so the duplicate error event of a deduped tool
    /// call is a no-op.
    pub async fn on_llm_error(&self, span: &TraceSpan, error: &Error) {
        self.mark_error_and_end(&span.invoke_id, error);
    }

    /// Derives the parent [`Context`] of a frame from its `parent_invoke_id`.
    ///
    /// Python counterpart: `_resolve_parent_context`, which looks the parent
    /// state up in the span manager and turns it into an OTel context. `None`
    /// (no parent id, or no open span for it) makes the new span a trace root.
    #[must_use]
    fn resolve_parent_context(&self, trace_span: &TraceSpan) -> Option<Context> {
        self.span_manager
            .parent_context(trace_span.parent_invoke_id.as_deref())
    }

    /// Starts a span, applies the base fields, and registers it.
    ///
    /// Python counterpart: `_start_and_push`. The state is pushed inside this
    /// helper, so callers continue with [`OtelAgentSpanManager::with_state`]
    /// instead of receiving it (the state owns the span and cannot be cloned).
    /// `recorded` is forced to `true` like Python's `state.recorded = True`: the
    /// SDK hides the sampling decision and attribute writes are no-ops on a
    /// dropped span anyway.
    fn start_and_push(&self, name: &str, kind: SpanKind, trace_span: &TraceSpan) {
        let parent_context = self.resolve_parent_context(trace_span).unwrap_or_default();
        let span = self
            .tracer
            .span_builder(name.to_owned())
            .with_kind(kind)
            .start_with_context(&self.tracer, &parent_context);
        let mut state = OtelSpanState::new(span, trace_span.invoke_id.clone(), true);
        attrs::apply_start_attributes(&mut state.span, trace_span);
        // The dedup probe reads the trace id back from the cache, because the
        // SDK span does not expose its attributes.
        state.record_attribute(OJ_TRACE_ID, trace_span.trace_id.clone());
        if trace_span.session_id.is_empty() {
            // Session fallback: a frame without a session id inherits the one
            // injected by the tracer registry (Python's `or self._session_id`).
            if let Some(session_id) = self
                .session_id
                .as_deref()
                .filter(|session_id| !session_id.is_empty())
            {
                state.record_attribute(OJ_SESSION_ID, session_id);
            }
        }
        self.span_manager.push(trace_span.invoke_id.clone(), state);
    }

    /// Closes the span of an invoke id and removes it from the manager.
    ///
    /// Python counterpart: `_end_and_pop`, including the `OK` status and the
    /// defensive no-op for an unknown invoke id.
    fn end_and_pop(&self, invoke_id: &str) {
        if let Some(mut state) = self.span_manager.pop(invoke_id) {
            state.span.set_status(Status::Ok);
            state.span.end();
        }
    }

    /// Writes the failure attributes of an invoke id and ends its span.
    ///
    /// Python counterpart: `_mark_error_and_end`. The attributes come from
    /// [`attrs::apply_error_attributes_with`], which covers the `ERROR` status,
    /// `openjiuwen.status`, the agent error message, the `openjiuwen.error`
    /// payload, `error.type`, and the `exception` event. An unknown invoke id is
    /// ignored.
    fn mark_error_and_end(&self, invoke_id: &str, error: &Error) {
        let Some(mut state) = self.span_manager.pop(invoke_id) else {
            return;
        };
        attrs::apply_error_attributes_with(
            &mut state.span,
            error.kind().as_str(),
            &error.message(),
            AttributeScope::Agent,
        );
        state.span.end();
    }

    /// Writes the agent identity / conversation attributes of a span.
    ///
    /// Python counterpart: `_set_agent_identity_attrs`. The facts come from the
    /// rail's common info (`agent_id`, `agent_name`, `agent_description`,
    /// `conversation_id`, `user_id`, `metadata`); an absent or falsy fact simply
    /// skips its attribute.
    fn set_agent_identity_attrs(state: &mut OtelSpanState, instance_info: &Value) {
        let Some(info) = instance_info.as_object() else {
            return;
        };
        for (source, key) in [
            ("agent_id", GEN_AI_AGENT_ID),
            ("agent_name", GEN_AI_AGENT_NAME),
            ("agent_description", GEN_AI_AGENT_DESCRIPTION),
            ("conversation_id", GEN_AI_CONVERSATION_ID),
            ("user_id", OJ_GEN_AI_USER_ID),
        ] {
            if let Some(text) = truthy_text(info.get(source)) {
                state.record_attribute(key, text);
            }
        }
        if let Some(metadata) = info.get("metadata").filter(|value| is_truthy(value)) {
            state.record_attribute(OJ_GEN_AI_METADATA, attrs::serialize_value(metadata));
        }
    }

    /// Writes the common `openjiuwen.agent.*` attributes of a non-LLM span.
    ///
    /// Python counterpart: `_set_non_llm_attrs` — the invoke type (the frame's
    /// own or the family default), the entity name (the frame's own or the
    /// forwarded `class_name`), the trace display name (the executing agent,
    /// falling back to the entity name), and the identity facts.
    fn set_non_llm_attrs(
        state: &mut OtelSpanState,
        trace_span: &TraceSpan,
        invoke_type: &str,
        instance_info: &Value,
    ) {
        let class_name = text_field(instance_info, "class_name").unwrap_or_default();
        let invoke_type_value = trace_span
            .invoke_type
            .map_or(invoke_type, |kind| kind.as_str());
        let name = if trace_span.name.is_empty() {
            class_name
        } else {
            trace_span.name.as_str()
        };
        state.record_attribute(OJ_AGENT_INVOKE_TYPE, invoke_type_value);
        state.record_attribute(OJ_AGENT_NAME, name);
        let trace_name =
            truthy_text(instance_info.get("agent_name")).unwrap_or_else(|| name.to_owned());
        state.record_attribute(OJ_GEN_AI_TRACE_NAME, trace_name);
        Self::set_agent_identity_attrs(state, instance_info);
    }

    /// Starts a non-LLM span, writes its attributes, and registers it.
    ///
    /// Python counterpart: `_start_non_llm_span`. The span name is the explicit
    /// override when the family supplies one, else `execute_tool {class_name}`
    /// for a tool event, else `{prefix}.{class_name}`.
    fn start_non_llm_span(
        &self,
        trace_span: &TraceSpan,
        inputs: &Value,
        instance_info: &Value,
        spec: &NonLlmSpan,
    ) {
        let name = non_llm_span_name(spec, instance_info);
        self.start_and_push(&name, SpanKind::Internal, trace_span);
        let redactor = Redactor::new(&self.config);
        self.span_manager
            .with_state(&trace_span.invoke_id, |state| {
                Self::set_non_llm_attrs(state, trace_span, spec.invoke_type, instance_info);
                for (key, value) in &spec.extra_attrs {
                    state.record_attribute(key, value.as_str());
                }
                if !inputs.is_null() {
                    state.record_attribute(
                        OJ_AGENT_INPUTS,
                        redactor.redact(Some(inputs), Some(RedactionField::Prompts)),
                    );
                }
            });
    }

    /// Closes the span of a successful non-LLM invocation.
    ///
    /// Python counterpart: `_end_non_llm_span` — the redacted outputs
    /// (`openjiuwen.agent.outputs`) and the end-time fields for a recording
    /// span, then the `OK` span. A frame with no open span is ignored, which
    /// makes the end event of a deduped tool call a no-op.
    fn end_non_llm_span(&self, trace_span: &TraceSpan, outputs: &Value) {
        let redactor = Redactor::new(&self.config);
        self.span_manager
            .with_state(&trace_span.invoke_id, |state| {
                if !state.recorded {
                    return;
                }
                if !outputs.is_null() {
                    state.record_attribute(
                        OJ_AGENT_OUTPUTS,
                        redactor.redact(Some(outputs), Some(RedactionField::Completions)),
                    );
                }
                set_end_attrs(state, trace_span);
            });
        self.end_and_pop(&trace_span.invoke_id);
    }

    /// Closes the span of a failed non-LLM invocation.
    ///
    /// Python counterpart: `_error_non_llm_span`, which is a thin alias of
    /// `_mark_error_and_end`; the error text lands on the agent scope's
    /// `openjiuwen.agent.error_message` for every family.
    fn error_non_llm_span(&self, trace_span: &TraceSpan, error: &Error) {
        self.mark_error_and_end(&trace_span.invoke_id, error);
    }

    /// Reports whether a plugin frame duplicates an open rail tool span.
    ///
    /// Python counterpart: `_is_duplicate_tool_event` (`bug #2046`). With the
    /// tool rail registered, one physical tool call emits two plugin events: the
    /// rail's (`type="plugin"`, carrying the tool-call id) and the legacy
    /// session-tracer decorator's (`type="tool"`, no id), fired strictly nested
    /// inside it. The decorator event is the duplicate and is dropped while its
    /// rail twin is still open.
    #[must_use]
    fn is_duplicate_tool_event(&self, trace_span: &TraceSpan, instance_info: &Value) -> bool {
        // Decorator fingerprint; rail events carry type="plugin" plus the id.
        if text_field(instance_info, "type") != Some(DECORATOR_PLUGIN_EVENT) {
            return false;
        }
        if instance_info.get("tool_call_id").is_some_and(is_truthy) {
            return false;
        }
        let tool_name = text_field(instance_info, "class_name")
            .filter(|name| !name.is_empty())
            .unwrap_or(trace_span.name.as_str());
        if tool_name.is_empty() {
            return false;
        }
        self.span_manager
            .has_open_tool_span(tool_name, &self.trace_id)
    }

    /// Runs `f` against the open state of an invoke id when it records.
    ///
    /// Python counterpart: the `state is None or not state.recorded` guard every
    /// family-specific attribute block starts with.
    fn with_recorded_state(&self, invoke_id: &str, f: impl FnOnce(&mut OtelSpanState)) {
        self.span_manager.with_state(invoke_id, |state| {
            if state.recorded {
                f(state);
            }
        });
    }
}

/// Builds the span name of a non-LLM frame.
///
/// Python counterpart: the `name` argument of `_start_non_llm_span` — the
/// explicit `span_name` when given, else the GenAI `execute_tool {tool}` rule
/// for a tool event, else `{prefix}.{class_name}`.
fn non_llm_span_name(spec: &NonLlmSpan, instance_info: &Value) -> String {
    if let Some(name) = spec.span_name.as_deref() {
        return name.to_owned();
    }
    let class_name = text_field(instance_info, "class_name").unwrap_or(UNKNOWN_CLASS_NAME);
    let is_tool = spec.extra_attrs.iter().any(|(key, value)| {
        *key == GEN_AI_OPERATION_NAME && value == GEN_AI_OPERATION_EXECUTE_TOOL
    });
    if is_tool {
        format!("execute_tool {class_name}")
    } else {
        format!("{}.{class_name}", spec.prefix)
    }
}

/// Writes the end-time fields of a span.
///
/// Python counterpart: `_set_end_attrs`, which `handler.py` calls before
/// `_end_and_pop`; [`attrs::apply_end_attributes`] covers both the fields and
/// the `OK` status.
fn set_end_attrs(state: &mut OtelSpanState, trace_span: &TraceSpan) {
    attrs::apply_end_attributes(&mut state.span, trace_span);
}

/// Returns a non-empty string field of an instance-info object.
///
/// Python counterpart: the `info.get(key)` lookups of the handler, which treat
/// an empty string as absent (`or` / truthiness checks). A `null`, a non-object
/// payload, or a non-string value yields `None`.
fn text_field<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value
        .as_object()?
        .get(key)?
        .as_str()
        .filter(|text| !text.is_empty())
}

/// Renders a truthy attribute value as text, `None` for a falsy one.
///
/// Python counterpart: `if value: ... str(value)`. [`attrs::serialize_value`]
/// supplies the text, so strings stay verbatim and other scalars use their JSON
/// spelling (`true`, `12`).
fn truthy_text(value: Option<&Value>) -> Option<String> {
    let value = value.filter(|inner| is_truthy(inner))?;
    Some(attrs::serialize_value(value))
}

/// Reports whether a JSON value is truthy in Python's `if value:` sense.
///
/// `null`, `false`, `0`, the empty string, and empty containers are falsy;
/// everything else is truthy. The handler uses the check for its
/// `instance_info` lookups, exactly like Python's optional-forwarding guards.
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
