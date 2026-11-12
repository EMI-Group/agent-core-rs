//! Workflow-dimension `OTel` handler: workflow frames become a span tree.
//!
//! Port of `OtelWorkflowHandler` from
//! `openjiuwen/extensions/tracer_otel/handler.py` (v0.1.20). The handler maps
//! the session tracer's workflow frames onto `OTel` spans with Python's names,
//! kinds, attributes, statuses, and parents:
//!
//! - one *layer root* per workflow: span name = its invoke id,
//!   [`SpanKind::Internal`];
//! - one *component* span per workflow node: `chat {model}`
//!   ([`SpanKind::Client`]), `execute_tool {name}`, `retrieval {name}`, or
//!   `component.{invoke_id}`;
//! - sub-workflow roots and the nodes of a sub-workflow parent onto the host
//!   component span.
//!
//! Three bookkeeping maps drive the tree, exactly as in Python: the span
//! manager (`invoke_id → OtelSpanState`, owned by [`OtelWorkflowSpanManager`]),
//! the layer roots (`node_id → root span`), and the components (`invoke_id →
//! component span`). Python stores whole `OtelSpanState` objects in the latter
//! two; [`OtelSpanState`] owns its span and is not `Clone`, so this handler
//! stores a lightweight [`SpanRef`] (invoke id, span context, workflow id,
//! registration order) and resolves the live state through the manager.
//!
//! # Mapping decisions
//!
//! Python's callbacks take keyword arguments that Rust trace frames do not
//! carry, so they are folded into `metadata`:
//!
//! - `parent_node_id` (string, default `""`), `session_id`, and `source_ids`
//!   (array) are read from `metadata` instead of keyword arguments;
//! - `metadata` is the only source of workflow/component facts, so
//!   [`OtelWorkflowHandler::on_call_start`] applies every workflow attribute and
//!   [`OtelWorkflowHandler::on_pre_invoke`] only refreshes the input-derived
//!   ones (Python re-applies `component_metadata` there);
//! - Python's exception branch of `on_invoke` is split in two, because a Rust
//!   trace frame carries no exception object:
//!   [`OtelWorkflowHandler::on_call_error`] for failures and
//!   [`OtelWorkflowHandler::on_call_interrupted`] for control-flow interrupts.
//!
//! # Parent resolution
//!
//! Python picks the parent from the layer/component maps in this order: a
//! workflow root with an empty `parent_node_id` reuses the cached root context,
//! starts a new trace when the `workflow_id` changed, otherwise parents onto the
//! *last inserted* component span (falling back to the layer root); `""` for a
//! component means the root workflow's layer root; a non-empty parent id is
//! looked up in the component map and then in the layer-root map. Python reads
//! "last inserted" from the insertion order of its dict; `BTreeMap` iterates by
//! key, so the insertion order is kept explicitly in `SpanRef::sequence` and the
//! highest sequence wins.
//!
//! # Deviations from Python
//!
//! - No ambient-context attach/detach: Rust has no `otel_context.attach` token on
//!   the session path, so the handler keeps no token and parents children
//!   through `SpanRef::parent_context` (a remote-parent [`Context`]).
//! - [`OtelWorkflowHandler::on_call_error`] and
//!   [`OtelWorkflowHandler::on_call_interrupted`] take the failure/interrupt
//!   payload directly instead of inspecting an exception. Both remove the ended
//!   invoke id from the layer-root map (Python's error branch leaves the closed
//!   span in `_layer_root_spans`), so a later round never parents onto a closed
//!   span; every other layer root is preserved for multi-round continuity.
//! - `on_invoke` writes `openjiuwen.inner_error` on the non-error path when the
//!   payload carries it; Python writes it only in the exception branch.
//! - The inputs payload is cached at `on_call_start` as well as at
//!   `on_pre_invoke`, so `on_call_done` re-serializes the post-transform inputs
//!   even when no `on_pre_invoke` frame arrived. The payload is cloned because
//!   Rust owns its values and cannot observe an in-place mutation.
//! - Fail-open: the callbacks that perform fallible work (buffer serialization
//!   in the terminal callbacks) run through a private `Result` body and log
//!   failures at `warn`; the remaining callbacks are infallible by construction
//!   (no `unwrap` on payload data, no panics) and log anomalies inline.
//! - Scalar metadata is rendered with its JSON text (`true`, `12`) where Python
//!   would `str()` it; component metadata carries strings in practice.
//!
//! [`SpanKind::Internal`]: opentelemetry::trace::SpanKind::Internal
//! [`SpanKind::Client`]: opentelemetry::trace::SpanKind::Client

// The otel `mod.rs` re-exports of this module land in a follow-up commit owned by
// the parent agent, so nothing in the crate calls the handler yet and its `pub`
// items are not reachable from outside the module tree. The `unused_async` opt-out
// is deliberate as well: the callbacks mirror Python's coroutines and the
// session-callback contract, while their bodies stay fully synchronous so no lock
// is ever held across an `await`. All of these go away with the re-export.
#![allow(
    dead_code,
    unreachable_pub,
    clippy::unused_async,
    clippy::unused_async_trait_impl
)]

mod spans;
#[cfg(test)]
mod tests;

use std::collections::BTreeSet;
use std::sync::Mutex;

use opentelemetry::trace::{Span as _, Status, Tracer as _};
use opentelemetry::KeyValue;
use opentelemetry_sdk::trace::SdkTracer;
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};

use super::attrs::{self, AttributeScope};
use super::config::OtelTracerConfig;
use super::handle::TracerHandle;
use super::redaction::Redactor;
use super::semconv::{
    GEN_AI_DATA_SOURCE_ID, GEN_AI_OPERATION_CHAT, GEN_AI_OPERATION_EXECUTE_TOOL,
    GEN_AI_OPERATION_NAME, GEN_AI_OPERATION_RETRIEVAL, GEN_AI_REQUEST_MODEL,
    GEN_AI_RETRIEVAL_QUERY_TEXT, GEN_AI_RETRIEVAL_TOP_K, GEN_AI_TOOL_NAME, OJ_INNER_ERROR,
    OJ_INTERACTIVE_INPUTS, OJ_INVOKE_ID, OJ_PARENT_NODE_ID, OJ_SESSION_ID, OJ_SOURCE_IDS,
    OJ_START_TIME, OJ_STATUS, OJ_STREAM_INPUTS, OJ_STREAM_OUTPUTS, OJ_TRACE_ID,
    OJ_WORKFLOW_ERROR_MESSAGE, OJ_WORKFLOW_INPUTS, OJ_WORKFLOW_INVOKE_DATA, OJ_WORKFLOW_OUTPUTS,
    SPAN_STATUS_FINISH, SPAN_STATUS_INTERRUPTED,
};
use super::span_manager::{OtelSpanState, OtelWorkflowSpanManager};

use spans::{
    is_retrieval_state, lock, record_documents, retrieval_query, set_end_attributes,
    set_workflow_attributes, string_array, truthy_text, CallFacts, HandlerState, SpanFamily,
    SpanRef,
};

/// `tracing` target of every warning this module emits.
const LOG_TARGET: &str = "openjiuwen_extensions::otel::workflow_handler";

/// Workflow-dimension handler: turns workflow frames into `OTel` spans.
///
/// Build one per tracer session with [`Self::new`] or from an initialized
/// [`TracerHandle`] with [`Self::from_handle`], then feed it the workflow
/// callbacks of the session tracer (`on_call_start`, `on_pre_invoke`,
/// `on_invoke`, `on_call_done`, …). Every callback is fail-open: failures are
/// logged at `warn` and never reach the caller.
///
/// The handler is `Send + Sync`, so one instance can serve frames from several
/// tasks; the shared state and the span manager both sit behind mutexes that are
/// never held across an `await`.
#[derive(Debug)]
pub struct OtelWorkflowHandler {
    /// Tracer bound to the private provider.
    tracer: SdkTracer,
    /// Configuration driving redaction and field overrides.
    config: OtelTracerConfig,
    /// Open spans plus the incremental buffers.
    manager: OtelWorkflowSpanManager,
    /// Trace/session ids and the layer-root/component maps.
    state: Mutex<HandlerState>,
}

impl OtelWorkflowHandler {
    /// Builds a handler over `tracer`.
    ///
    /// `trace_id` is written to `openjiuwen.trace.id`, like Python's optional
    /// `trace_id` constructor argument; `None` starts with an empty id, which
    /// [`Self::set_trace_id`] fills in later.
    #[must_use]
    pub fn new(tracer: SdkTracer, config: OtelTracerConfig, trace_id: Option<String>) -> Self {
        Self {
            tracer,
            config,
            manager: OtelWorkflowSpanManager::new(),
            state: Mutex::new(HandlerState {
                trace_id: trace_id.unwrap_or_default(),
                ..HandlerState::default()
            }),
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

    /// Sets the trace id and cleans up spans of a superseded execution.
    ///
    /// Python counterpart: `set_trace_id`. A changed non-empty id drops the
    /// cached root context and ends every span still registered in the
    /// layer-root/component maps, so the next frames belong to a fresh trace.
    pub fn set_trace_id(&mut self, trace_id: impl Into<String>) {
        let trace_id = trace_id.into();
        let mut state = lock(&self.state);
        if !trace_id.is_empty() && trace_id != state.trace_id {
            state.clear_cached_context();
            self.cleanup_stale_spans(&mut state);
        }
        state.trace_id = trace_id;
    }

    /// Sets the session id used when a frame carries none.
    ///
    /// Python counterpart: `set_session_id`, which also cleans up the spans of a
    /// previous session.
    pub fn set_session_id(&mut self, session_id: impl Into<String>) {
        let session_id = session_id.into();
        let mut state = lock(&self.state);
        if !session_id.is_empty() && state.cached_session_id.as_deref() != Some(session_id.as_str())
        {
            state.clear_cached_context();
            state.cached_session_id = Some(session_id.clone());
            self.cleanup_stale_spans(&mut state);
        }
        state.session_id = (!session_id.is_empty()).then_some(session_id);
    }

    /// Ends every span registered in the layer-root and component maps.
    ///
    /// Python counterpart: `_cleanup_stale_spans` (minus the ambient-context
    /// detach, which has no Rust equivalent). The spans are ended through the
    /// manager, deduplicated by invoke id because a root is registered twice, and
    /// the maps plus the cached root context are cleared afterwards. A ref
    /// without an open state is an anomaly, logged and skipped.
    fn cleanup_stale_spans(&self, state: &mut HandlerState) {
        let mut invoke_ids: BTreeSet<String> = BTreeSet::new();
        invoke_ids.extend(
            state
                .layer_roots
                .values()
                .map(|span| span.invoke_id.clone()),
        );
        invoke_ids.extend(state.components.values().map(|span| span.invoke_id.clone()));
        for invoke_id in invoke_ids {
            if let Some(mut open) = self.manager.pop(&invoke_id) {
                open.span.end();
            } else {
                tracing::warn!(
                    target: LOG_TARGET,
                    %invoke_id,
                    "otel workflow handler: cleanup stale span failed: no open span state"
                );
            }
        }
        state.layer_roots.clear();
        state.components.clear();
    }

    /// Starts the span of a workflow or component invocation.
    ///
    /// Python counterpart: `on_call_start`. `parent_node_id`, `session_id`, and
    /// `source_ids` arrive through `metadata` (see the module docs). A
    /// non-recording span is still pushed and registered so the lifecycle stays
    /// symmetric, but no attribute is serialized for it.
    pub async fn on_call_start(&self, invoke_id: &str, metadata: &Value, inputs: Option<&Value>) {
        // Python replaces a `None` metadata with `{}` before every lookup, so a
        // null payload behaves exactly like an empty object.
        let metadata = if metadata.is_object() {
            metadata
        } else {
            &Value::Null
        };
        let mut facts = CallFacts::from_metadata(metadata);
        let mut handler = lock(&self.state);
        facts.trace_id.clone_from(&handler.trace_id);
        if facts.session_id.is_none() {
            facts.session_id.clone_from(&handler.session_id);
        }
        let parent_context = handler
            .resolve_parent_context(&facts.parent_node_id, metadata)
            .unwrap_or_default();
        let span = self
            .tracer
            .span_builder(facts.span_name(invoke_id))
            .with_kind(facts.span_kind())
            .start_with_context(&self.tracer, &parent_context);
        let mut state = OtelSpanState::from_span(span, invoke_id);
        state.workflow_id.clone_from(&facts.workflow_id);
        if state.recorded {
            self.write_attributes(&facts, &mut state, metadata, inputs);
        }
        let span_ref = SpanRef {
            invoke_id: invoke_id.to_owned(),
            span_context: state.span_context(),
            workflow_id: facts.workflow_id.clone(),
            sequence: handler.next_sequence(),
        };
        self.manager.push(invoke_id, state);
        if facts.is_root() {
            // Registered twice: under the parent node id (the key children of the
            // root workflow resolve) and under its own invoke id (the key a
            // sub-workflow root resolves).
            handler
                .layer_roots
                .insert(facts.parent_node_id.clone(), span_ref.clone());
            handler
                .layer_roots
                .insert(invoke_id.to_owned(), span_ref.clone());
            handler.cache_root(&span_ref, &facts.trace_id);
        } else {
            handler.components.insert(invoke_id.to_owned(), span_ref);
        }
    }

    /// Writes the start attributes of a recording span.
    ///
    /// Python counterpart: the base attributes, the `gen_ai.*` component
    /// branches, `_set_workflow_attrs`, and the workflow-inputs/retrieval-query
    /// block of `on_call_start`.
    fn write_attributes(
        &self,
        facts: &CallFacts,
        state: &mut OtelSpanState,
        metadata: &Value,
        inputs: Option<&Value>,
    ) {
        let redactor = Redactor::new(&self.config);
        let invoke_id = state.invoke_id.clone();
        state.record_attribute(OJ_TRACE_ID, facts.trace_id.as_str());
        if let Some(session_id) = facts.session_id.as_deref().filter(|id| !id.is_empty()) {
            state.record_attribute(OJ_SESSION_ID, session_id);
        }
        state.record_attribute(OJ_INVOKE_ID, invoke_id.as_str());
        state.record_attribute(OJ_PARENT_NODE_ID, facts.parent_node_id.as_str());
        state.record_attribute(OJ_START_TIME, attrs::local_timestamp());
        if let Some(source_ids) = metadata.get("source_ids").filter(|value| !value.is_null()) {
            state.record_attribute(OJ_SOURCE_IDS, attrs::serialize_value(source_ids));
        }
        match facts.family {
            SpanFamily::Llm => {
                state.record_attribute(GEN_AI_OPERATION_NAME, GEN_AI_OPERATION_CHAT);
                if let Some(model) = facts.model() {
                    state.record_attribute(GEN_AI_REQUEST_MODEL, model);
                }
            }
            SpanFamily::Tool => {
                state.record_attribute(GEN_AI_OPERATION_NAME, GEN_AI_OPERATION_EXECUTE_TOOL);
                state.record_attribute(GEN_AI_TOOL_NAME, facts.name(&invoke_id));
            }
            SpanFamily::Retrieval => {
                state.record_attribute(GEN_AI_OPERATION_NAME, GEN_AI_OPERATION_RETRIEVAL);
                if let Some(top_k) = metadata.get("top_k").and_then(Value::as_i64) {
                    state
                        .span
                        .set_attribute(KeyValue::new(GEN_AI_RETRIEVAL_TOP_K, top_k));
                }
                if let Some(ids) = metadata
                    .get("data_source_ids")
                    .and_then(Value::as_array)
                    .filter(|ids| !ids.is_empty())
                {
                    let ids = ids.iter().map(attrs::serialize_value).collect::<Vec<_>>();
                    state
                        .span
                        .set_attribute(KeyValue::new(GEN_AI_DATA_SOURCE_ID, string_array(ids)));
                }
                if let Some(model) = truthy_text(metadata.get("request_model")) {
                    state.record_attribute(GEN_AI_REQUEST_MODEL, model);
                }
            }
            SpanFamily::Root | SpanFamily::Component => {}
        }
        set_workflow_attributes(state, metadata, &invoke_id);
        if let Some(inputs) = inputs.filter(|value| !value.is_null()) {
            state.record_attribute(OJ_WORKFLOW_INPUTS, redactor.redact(Some(inputs), None));
            // Cached for the inputs refresh; Python only caches it in
            // `on_pre_invoke` (see the module docs).
            state.inputs = Some(inputs.clone());
            if facts.is_retrieval() {
                if let Some(query) = retrieval_query(inputs) {
                    state.record_attribute(
                        GEN_AI_RETRIEVAL_QUERY_TEXT,
                        redactor.redact(Some(&Value::String(query)), None),
                    );
                }
            }
        }
    }

    /// Closes the span of a successful invocation.
    ///
    /// Python counterpart: `on_call_done` — refresh the inputs, flush the
    /// buffers, then end the span with the outputs, retrieval documents, the
    /// end-time fields, `openjiuwen.status = finish`, and an OK status.
    pub async fn on_call_done(&self, invoke_id: &str, outputs: Option<&Value>) {
        if let Err(error) = self.finish_call(invoke_id, outputs) {
            tracing::warn!(
                target: LOG_TARGET,
                error = %error,
                %invoke_id,
                "otel workflow handler: on_call_done failed"
            );
        }
    }

    /// Implements [`Self::on_call_done`].
    fn finish_call(&self, invoke_id: &str, outputs: Option<&Value>) -> Result<()> {
        if self.state_is_recorded(invoke_id) {
            self.refresh_inputs(invoke_id);
            self.flush_buffered_data(invoke_id)?;
        }
        let Some(mut state) = self.manager.pop(invoke_id) else {
            return Ok(());
        };
        if state.recorded {
            if let Some(outputs) = outputs.filter(|value| !value.is_null()) {
                let redactor = Redactor::new(&self.config);
                state.record_attribute(OJ_WORKFLOW_OUTPUTS, redactor.redact(Some(outputs), None));
                if is_retrieval_state(&state) {
                    record_documents(&mut state, outputs, redactor);
                }
            }
            set_end_attributes(&mut state);
            state.record_attribute(OJ_STATUS, SPAN_STATUS_FINISH);
            state.span.set_status(Status::Ok);
        }
        state.span.end();
        self.forget_span(invoke_id);
        Ok(())
    }

    /// Refreshes the input-derived attributes of an invocation.
    ///
    /// Python counterpart: `on_pre_invoke`'s inputs branch (its
    /// `component_metadata` argument has no Rust equivalent, because
    /// `on_call_start` already applied those attributes). The cached inputs are
    /// re-serialized after the transform callbacks had their chance to rewrite
    /// them, and a retrieval component also gets a fresh query text.
    pub async fn on_pre_invoke(&self, invoke_id: &str, inputs: Option<&Value>) {
        if !self.state_is_recorded(invoke_id) {
            return;
        }
        let Some(inputs) = inputs.filter(|value| !value.is_null()) else {
            return;
        };
        let redactor = Redactor::new(&self.config);
        self.manager.with_state(invoke_id, |state| {
            state.inputs = Some(inputs.clone());
            state.record_attribute(OJ_WORKFLOW_INPUTS, redactor.redact(Some(inputs), None));
            if is_retrieval_state(state) {
                if let Some(query) = retrieval_query(inputs) {
                    state.record_attribute(
                        GEN_AI_RETRIEVAL_QUERY_TEXT,
                        redactor.redact(Some(&Value::String(query)), None),
                    );
                }
            }
        });
    }

    /// Buffers one stream-input chunk.
    ///
    /// Python counterpart: `on_pre_stream`, which only buffers object chunks; the
    /// buffer is flushed as one attribute by [`Self::on_call_done`].
    pub async fn on_pre_stream(&self, invoke_id: &str, chunk: &Value) {
        if chunk.is_object() {
            self.manager.append_stream_input(invoke_id, chunk.clone());
        }
    }

    /// Buffers one intermediate invocation payload.
    ///
    /// Python counterpart: the non-exception branch of `on_invoke` — the payload
    /// is buffered for the flush and the inputs attribute is refreshed. As a
    /// documented deviation, a payload carrying `inner_error` also records it
    /// right away.
    pub async fn on_invoke(&self, invoke_id: &str, data: &Value) {
        if !data.is_null() {
            self.manager.append_on_invoke_data(invoke_id, data.clone());
            if let Some(inner_error) = data
                .as_object()
                .and_then(|payload| payload.get("inner_error"))
            {
                let text = attrs::serialize_value(inner_error);
                self.manager.with_state(invoke_id, |state| {
                    state.record_attribute(OJ_INNER_ERROR, text);
                });
            }
        }
        self.refresh_inputs(invoke_id);
    }

    /// Records the outputs of a still open invocation.
    ///
    /// Python counterpart: `on_post_invoke`, which projects the redacted outputs
    /// (and the retrieval documents) onto the span before it ends.
    pub async fn on_post_invoke(&self, invoke_id: &str, outputs: &Value) {
        if !self.state_is_recorded(invoke_id) {
            return;
        }
        let Some(outputs) = Some(outputs).filter(|value| !value.is_null()) else {
            return;
        };
        let redactor = Redactor::new(&self.config);
        self.manager.with_state(invoke_id, |state| {
            state.record_attribute(OJ_WORKFLOW_OUTPUTS, redactor.redact(Some(outputs), None));
            if is_retrieval_state(state) {
                record_documents(state, outputs, redactor);
            }
        });
    }

    /// Buffers one stream-output chunk.
    ///
    /// Python counterpart: `on_post_stream`, which only buffers object chunks.
    pub async fn on_post_stream(&self, invoke_id: &str, chunk: &Value) {
        if chunk.is_object() {
            self.manager.append_stream_output(invoke_id, chunk.clone());
        }
    }

    /// Records the interactive (human-in-the-loop) inputs of an invocation.
    ///
    /// Python counterpart: `on_interact`, which writes the redacted inputs to
    /// `openjiuwen.interactive_inputs`.
    pub async fn on_interact(&self, invoke_id: &str, inputs: &Value) {
        if !self.state_is_recorded(invoke_id) {
            return;
        }
        let Some(inputs) = Some(inputs).filter(|value| !value.is_null()) else {
            return;
        };
        let redactor = Redactor::new(&self.config);
        self.manager.with_state(invoke_id, |state| {
            state.record_attribute(OJ_INTERACTIVE_INPUTS, redactor.redact(Some(inputs), None));
        });
    }

    /// Ends an invocation that failed.
    ///
    /// Python counterpart: the exception branch of `on_invoke` for a real failure
    /// — an `ERROR` status, `openjiuwen.status = error`, the error message, the
    /// `openjiuwen.error` payload keyed by
    /// [`openjiuwen_common::error::ErrorKind`], the `error.type`, and the
    /// `exception` event. The end-time fields are written, the span is ended, and
    /// any component span still open under the same invoke id is finished. Layer
    /// roots are preserved for multi-round context; only the ended invoke id is
    /// dropped from the maps.
    pub async fn on_call_error(&self, invoke_id: &str, error: &Error) {
        if let Err(failure) = self.fail_call(invoke_id, error) {
            tracing::warn!(
                target: LOG_TARGET,
                error = %failure,
                %invoke_id,
                "otel workflow handler: on_call_error failed"
            );
        }
    }

    /// Implements [`Self::on_call_error`].
    fn fail_call(&self, invoke_id: &str, error: &Error) -> Result<()> {
        self.refresh_inputs(invoke_id);
        self.flush_buffered_data(invoke_id)?;
        let inner_error = self.buffered_inner_error(invoke_id);
        let Some(mut state) = self.manager.pop(invoke_id) else {
            return Ok(());
        };
        attrs::apply_error_attributes_with(
            &mut state.span,
            error.kind().as_str(),
            &error.message(),
            AttributeScope::Workflow,
        );
        if let Some(inner_error) = inner_error {
            state.record_attribute(OJ_INNER_ERROR, inner_error);
        }
        set_end_attributes(&mut state);
        state.span.end();
        self.cleanup_components(invoke_id);
        self.forget_span(invoke_id);
        Ok(())
    }

    /// Ends an invocation that was interrupted.
    ///
    /// Python counterpart: the `GraphInterrupt` branch of `on_invoke`. An
    /// interrupt is control flow, not a failure: `openjiuwen.status` becomes
    /// `interrupted`, the reason lands on
    /// `openjiuwen.workflow.error_message`, and the span keeps a non-error status
    /// and no `openjiuwen.error`. The lifecycle matches [`Self::on_call_error`].
    pub async fn on_call_interrupted(&self, invoke_id: &str, reason: Option<&str>) {
        if let Err(failure) = self.interrupt_call(invoke_id, reason) {
            tracing::warn!(
                target: LOG_TARGET,
                error = %failure,
                %invoke_id,
                "otel workflow handler: on_call_interrupted failed"
            );
        }
    }

    /// Implements [`Self::on_call_interrupted`].
    fn interrupt_call(&self, invoke_id: &str, reason: Option<&str>) -> Result<()> {
        self.refresh_inputs(invoke_id);
        self.flush_buffered_data(invoke_id)?;
        let inner_error = self.buffered_inner_error(invoke_id);
        let Some(mut state) = self.manager.pop(invoke_id) else {
            return Ok(());
        };
        attrs::set_text_attribute(&mut state.span, OJ_STATUS, SPAN_STATUS_INTERRUPTED);
        if let Some(reason) = reason.filter(|text| !text.is_empty()) {
            attrs::set_text_attribute(&mut state.span, OJ_WORKFLOW_ERROR_MESSAGE, reason);
        }
        if let Some(inner_error) = inner_error {
            state.record_attribute(OJ_INNER_ERROR, inner_error);
        }
        set_end_attributes(&mut state);
        state.span.end();
        self.cleanup_components(invoke_id);
        self.forget_span(invoke_id);
        Ok(())
    }

    /// Whether the open state of an invoke id records attributes.
    fn state_is_recorded(&self, invoke_id: &str) -> bool {
        self.manager
            .with_state(invoke_id, |state| state.recorded)
            .unwrap_or(false)
    }

    /// Re-serializes the cached inputs of an invocation.
    ///
    /// Python counterpart: `_refresh_inputs_attribute` — a no-op without an open
    /// state or without cached inputs.
    fn refresh_inputs(&self, invoke_id: &str) {
        let redactor = Redactor::new(&self.config);
        self.manager.with_state(invoke_id, |state| {
            let Some(inputs) = state.inputs.clone() else {
                return;
            };
            state.record_attribute(OJ_WORKFLOW_INPUTS, redactor.redact(Some(&inputs), None));
        });
    }

    /// Writes the buffered payloads as one attribute each.
    ///
    /// Python counterpart: `_flush_buffered_data`, which only writes a non-empty
    /// buffer. The three buffer snapshots are taken *before* the manager is
    /// locked again: [`OtelWorkflowSpanManager::with_state`] holds that lock, so
    /// a buffer getter inside the closure would deadlock.
    fn flush_buffered_data(&self, invoke_id: &str) -> Result<()> {
        let mut attributes: Vec<(&'static str, String)> = Vec::with_capacity(3);
        let on_invoke_data = self.manager.on_invoke_data(invoke_id);
        if !on_invoke_data.is_empty() {
            attributes.push((OJ_WORKFLOW_INVOKE_DATA, serialize_buffer(&on_invoke_data)?));
        }
        let stream_inputs = self.manager.stream_inputs(invoke_id);
        if !stream_inputs.is_empty() {
            attributes.push((OJ_STREAM_INPUTS, serialize_buffer(&stream_inputs)?));
        }
        let stream_outputs = self.manager.stream_outputs(invoke_id);
        if !stream_outputs.is_empty() {
            attributes.push((OJ_STREAM_OUTPUTS, serialize_buffer(&stream_outputs)?));
        }
        if attributes.is_empty() {
            return Ok(());
        }
        self.manager.with_state(invoke_id, |state| {
            for (key, text) in attributes {
                state.record_attribute(key, text);
            }
        });
        Ok(())
    }

    /// The last buffered `inner_error` payload of an invocation.
    ///
    /// Python records `on_invoke_data["inner_error"]` in its exception branch;
    /// Rust frames carry the payload through the buffer instead, so the terminal
    /// callbacks read it back from there.
    fn buffered_inner_error(&self, invoke_id: &str) -> Option<String> {
        self.manager
            .on_invoke_data(invoke_id)
            .iter()
            .rev()
            .find_map(|data| {
                data.as_object()?
                    .get("inner_error")
                    .map(attrs::serialize_value)
            })
    }

    /// Removes the ended invoke id from the layer-root and component maps.
    ///
    /// Python counterpart: the tail of `on_call_done` — the invoke id leaves both
    /// maps, and the `""` layer root is dropped when it is the span that just
    /// ended.
    fn forget_span(&self, invoke_id: &str) {
        let mut handler = lock(&self.state);
        handler.components.remove(invoke_id);
        handler.layer_roots.remove(invoke_id);
        if handler
            .layer_roots
            .get("")
            .is_some_and(|root| root.invoke_id == invoke_id)
        {
            handler.layer_roots.remove("");
        }
    }

    /// Finishes the component spans registered under `invoke_id`.
    ///
    /// Python counterpart: the defensive loop of `on_invoke`'s exception branch,
    /// which ends a component span still open when its workflow failed. The state
    /// is popped from the manager as well, so no half-cleaned entry stays
    /// reachable. The layer roots are left alone.
    fn cleanup_components(&self, invoke_id: &str) {
        let mut handler = lock(&self.state);
        let stale_keys = handler
            .components
            .iter()
            .filter(|(_, span)| span.invoke_id == invoke_id)
            .map(|(key, _)| key.clone())
            .collect::<Vec<_>>();
        for key in stale_keys {
            handler.components.remove(&key);
            if let Some(mut open) = self.manager.pop(&key) {
                open.span.end();
            } else {
                tracing::warn!(
                    target: LOG_TARGET,
                    %key,
                    "otel workflow handler: cleanup component span failed: no open span state"
                );
            }
        }
    }
}

/// Serializes a buffered payload list as compact JSON.
///
/// Python counterpart: `_serialize`, which `_flush_buffered_data` applies to the
/// whole buffer; an empty buffer is never serialized.
fn serialize_buffer(values: &[Value]) -> Result<String> {
    serde_json::to_string(values).map_err(|error| {
        Error::validation(format!("cannot serialize a workflow trace buffer: {error}"))
    })
}
