//! [`TraceSink`] implementation mapping session trace frames onto `OTel` spans.
//!
//! Port of the frame-level seam of
//! `openjiuwen/extensions/tracer_otel/handler.py` (v0.1.20). Python registers
//! two async callback families (`OtelAgentHandler`, `OtelWorkflowHandler`) with
//! the session tracer registry; Rust sessions emit one synchronous
//! [`TraceFrame`] per lifecycle transition instead, so this sink is the adapter
//! between the two models.
//!
//! # Frame-model adaptation
//!
//! A frame carries a full [`TraceSpan`] snapshot plus its [`SpanStatus`], so the
//! sink derives the Python callback from the frame rather than from a method
//! name:
//!
//! | Frame | Python counterpart |
//! |---|---|
//! | agent `start` | `on_llm_start` / `on_plugin_start` / `on_retriever_start` / `on_prompt_start` / `on_chain_start` / `on_evaluator_start` / `on_workflow_start` |
//! | agent `running` | `on_llm_request` (the only mid-flight agent event the frame model preserves) |
//! | agent `finish` | `on_llm_end` / `on_*_end` |
//! | agent `error` | `on_*_error` |
//! | agent `interrupted` | the interrupt branch of the agent lifecycle (no Python callback) |
//! | workflow / component `start` | `on_call_start` |
//! | workflow / component `running` | `on_pre_invoke` |
//! | workflow / component `finish` | `on_call_done` (plus `on_post_invoke`'s outputs) |
//! | workflow / component `error` | the failure branch of `on_invoke` |
//! | workflow / component `interrupted` | the `GraphInterrupt` branch of `on_invoke` |
//!
//! `frame.item_type` is ignored: the [`TraceSpanKind`] of the payload already
//! selects the agent or workflow path, exactly as Python's two handler classes
//! do.
//!
//! ## `meta_data` → workflow buffers
//!
//! The workflow handler keeps three incremental buffers that Python fills from
//! dedicated callbacks (`on_invoke`, `on_pre_stream`, `on_post_stream`) and
//! flushes as one JSON attribute each on `on_call_done`. Rust trace frames have
//! no per-chunk callbacks, so the sink reads the buffered payloads out of
//! the span's `meta_data` map: the keys `on_invoke_data`, `stream_inputs`, and
//! `stream_outputs` feed the matching buffer (an array value contributes each
//! element, any other value contributes itself), and the terminal frame flushes
//! the three snapshots as `openjiuwen.workflow.invoke_data`,
//! `openjiuwen.stream_inputs`, and `openjiuwen.stream_outputs`.
//!
//! # Lazy open
//!
//! The session tracer emits exactly one frame per span — its terminal
//! `SpanHandle::complete` — so the sink cannot rely on a `start` frame. An
//! agent, workflow, or component `running`/terminal frame whose invoke id has no
//! open state first *opens* the span: the frame's `start_time` becomes the
//! `OTel` start timestamp, and the span gets the same name, kind, operation,
//! parent, and identity attributes a `start` frame would have written, after
//! which the running or terminal attributes are applied on top.
//!
//! A `running`-only frame leaves the span open for its terminal frame, and a
//! span opened this way ends at the frame's `end_time` (falling back to the
//! local clock when the end is absent or precedes the start), so an idiomatic
//! `Tracer::with_sink(..)` + `start_span(..)` + `finish(..)` run still exports
//! one span with a realistic duration. A span opened by a real `start` frame
//! ends at the local clock instead, exactly like the typed handlers.
//!
//! # Provider lifetime
//!
//! [`TracerHandle`] is borrowed for the duration of [`OtelTraceSink::new`] and
//! is dropped immediately by [`OtelTraceSink::with_session`], so the sink keeps
//! its own clone of the tracer **and** of the private provider. The provider
//! clone pins the batch processor (and therefore span export) for as long as
//! the sink lives: without it, a dropped handle could shut the provider down
//! while spans the sink started are still open, losing their exported data.
//!
//! # Fail-open
//!
//! [`TraceSink::emit`] must never fail a traced run: it is synchronous, does no
//! I/O, never flushes, and never panics. A frame that cannot be mapped (a
//! payload of the wrong shape, or any frame while the sink holds no tracer) is
//! logged through `tracing` at `debug` and ignored; every other frame registers
//! or closes exactly one span, including a terminal frame for an invoke id the
//! sink has not seen yet, which lazily opens (see below). No lock is held across
//! a snapshot call that could re-enter a span manager.
//!
//! # Deviations from Python
//!
//! - No ambient-context attach/detach: Python attaches the new span to the
//!   `OTel` context and detaches it on pop. Rust has no token on the session
//!   path, so children resolve their parent through the span managers instead.
//! - The agent start path writes `openjiuwen.agent.inputs` for LLM frames too
//!   (Python only writes it for non-LLM spans); the LLM payload additionally
//!   gets `gen_ai.system_instructions` / `gen_ai.input.messages`.
//! - Frame spans carry no exception object, so a failure carries only the
//!   textual `error` field; the error code falls back to the shared
//!   `DEFAULT_ERROR_CODE` and `error.type` repeats it.
//! - Workflow spans reuse [`attrs::apply_start_attributes`], so they also carry
//!   `gen_ai.provider.name` and `openjiuwen.parent_invoke_id`, which Python's
//!   workflow handler does not write (it uses `openjiuwen.parent_node_id` only).
//! - The environment variable fallback for the session id is not needed: a
//!   frame always carries its own session id, and [`OtelTraceSink::with_session`]
//!   supplies a fallback for frames that carry an empty one.

use std::sync::Arc;

use opentelemetry_sdk::trace::{SdkTracer, SdkTracerProvider};

use openjiuwen_common::error::Result;
use openjiuwen_core::session::tracer::{TraceFrame, TraceSink, TraceSpanKind};

use super::config::{ExporterType, OtelTracerConfig};
use super::handle::{init_otel_tracer, TracerHandle};
use super::span_manager::{OtelAgentSpanManager, OtelWorkflowSpanManager};

mod frames;
#[cfg(test)]
mod tests;

/// `tracing` target of sink-level diagnostics.
const LOG_TARGET: &str = "openjiuwen_extensions::otel::sink";

/// Session [`TraceSink`] that turns trace frames into `OTel` spans.
///
/// One sink serves a whole session: it owns the agent and workflow span
/// managers, so child frames resolve their parents across dimensions, and it
/// keeps the private provider alive for its own lifetime (see the module docs).
#[derive(Debug)]
pub struct OtelTraceSink {
    /// Tracer bound to the private provider.
    tracer: Option<SdkTracer>,
    /// Clone of the private provider; see the module docs on provider lifetime.
    provider: Option<SdkTracerProvider>,
    /// Redaction and shaping configuration the spans are built from.
    config: OtelTracerConfig,
    /// `handle.is_active()` at construction; `false` for an inactive handle.
    recording: bool,
    /// Fallback session id for frames that carry an empty one.
    session_id: Option<String>,
    /// Open agent spans, keyed by invoke id.
    agent_spans: OtelAgentSpanManager,
    /// Open workflow/component spans plus their three buffers.
    workflow_spans: OtelWorkflowSpanManager,
}

impl OtelTraceSink {
    /// Builds a sink over a tracer handle.
    ///
    /// An inactive handle (no `otel` provider, or a shut-down one) yields a sink
    /// whose [`TraceSink::emit`] is a cheap no-op and whose
    /// [`Self::is_recording`] is `false`, so callers never branch on the
    /// feature.
    ///
    /// # Errors
    ///
    /// Reserved for future handle-level failures; building a sink from an
    /// already-initialized handle cannot fail today. The signature mirrors
    /// [`Self::with_session`].
    // The signature is fixed by the crate's public surface (a sink may need to
    // validate a caller-supplied handle later), so the `Result` stays even
    // though no branch returns `Err` today.
    #[allow(clippy::unnecessary_wraps)]
    pub fn new(handle: &TracerHandle) -> Result<Arc<Self>> {
        Ok(Arc::new(Self::from_handle(handle, None)))
    }

    /// Builds a sink over a private in-memory provider.
    ///
    /// Convenience entry point for hosts and tests that only need the frames to
    /// become spans: the provider is created internally (and kept alive by the
    /// sink), and `session_id` is used for frames that carry an empty
    /// `session_id`.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::validation`] when the default
    /// configuration is invalid (it is not), mirroring [`init_otel_tracer`].
    pub fn with_session(session_id: impl Into<String>) -> Result<Arc<Self>> {
        let config = OtelTracerConfig::default().with_exporter_type(ExporterType::Memory);
        let handle = init_otel_tracer(config)?;
        Ok(Arc::new(Self::from_handle(
            &handle,
            Some(session_id.into()),
        )))
    }

    /// Reports whether this sink records spans.
    ///
    /// `false` when the handle was inactive (no feature, or after
    /// [`TracerHandle::shutdown`]) or when it exposed no tracer/provider pair.
    #[must_use]
    pub fn is_recording(&self) -> bool {
        self.recording && self.tracer.is_some() && self.provider.is_some()
    }

    /// Builds the sink body, optionally with a fallback session id.
    fn from_handle(handle: &TracerHandle, session_id: Option<String>) -> Self {
        let tracer = handle.tracer();
        // The clone keeps the private provider (and its batch processor) alive
        // for as long as this sink lives; see the module docs.
        let provider = handle.provider().cloned();
        let recording = handle.is_active() && tracer.is_some() && provider.is_some();
        Self {
            tracer,
            provider,
            config: handle.config().clone(),
            recording,
            session_id,
            agent_spans: OtelAgentSpanManager::new(),
            workflow_spans: OtelWorkflowSpanManager::new(),
        }
    }

    /// The fallback session id of this sink, if it was built with one.
    fn fallback_session_id(&self) -> Option<&str> {
        self.session_id.as_deref().filter(|id| !id.is_empty())
    }
}

impl TraceSink for OtelTraceSink {
    /// Maps one frame onto the span lifecycle of its invoke id.
    ///
    /// Dispatch is on the payload's
    /// [`TraceSpanKind`](openjiuwen_core::session::tracer::TraceSpanKind) (agent
    /// versus workflow/component); `frame.item_type` is informational and
    /// ignored. Failures are logged and dropped, never propagated.
    fn emit(&self, frame: TraceFrame) {
        if !self.is_recording() {
            return;
        }
        match frame.payload.kind {
            TraceSpanKind::Agent => self.emit_agent(&frame.payload),
            TraceSpanKind::Workflow | TraceSpanKind::Component => {
                self.emit_workflow(&frame.payload);
            }
        }
    }
}
