//! OpenTelemetry tracing over the session [`TraceSink`] SPI.
//!
//! Ports `openjiuwen/extensions/tracer_otel/` (v0.1.20). Sessions emit
//! synchronous trace frames, so the primary seam is a sink ([`OtelTraceSink`])
//! plus typed handlers for callers that already hold a span.
//!
//! Everything that decides *what a span looks like* — configuration, semantic
//! conventions, redaction, span names — compiles with or without the `otel`
//! feature, so hosts can build configuration and assert naming rules without
//! pulling the SDK in. Without the feature, [`init_otel_tracer`] validates the
//! configuration and returns an inactive handle instead of failing.
//!
//! [`TraceSink`]: openjiuwen_core::session::tracer::TraceSink

pub mod config;
pub mod handle;
pub mod redaction;
pub mod semconv;
pub mod span_names;

#[cfg(feature = "otel")]
mod attrs;
#[cfg(feature = "otel")]
mod agent_handler;
#[cfg(feature = "otel")]
mod llm_attrs;
#[cfg(feature = "otel")]
pub mod provider;
#[cfg(feature = "otel")]
pub mod sink;
#[cfg(feature = "otel")]
pub mod span_manager;
#[cfg(feature = "otel")]
mod workflow_handler;
