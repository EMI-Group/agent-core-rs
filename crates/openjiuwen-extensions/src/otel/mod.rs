//! OpenTelemetry tracing over the session [`TraceSink`] SPI.
//!
//! Ports `openjiuwen/extensions/tracer_otel/` (v0.1.20). Sessions emit
//! synchronous trace frames, so the primary seam is a sink ([`OtelTraceSink`])
//! plus typed handlers ([`OtelAgentHandler`], [`OtelWorkflowHandler`]) for
//! callers that already hold a [`TraceSpan`].
//!
//! Everything that decides *what a span looks like* — configuration, semantic
//! conventions, redaction, span names — compiles with or without the `otel`
//! feature, so hosts can build configuration and assert naming rules without
//! pulling the SDK in. Without the feature, [`init_otel_tracer`] validates the
//! configuration and returns an inactive handle instead of failing.
//!
//! # Seam
//!
//! ```
//! use openjiuwen_extensions::otel::{init_otel_tracer, ExporterType, OtelTracerConfig};
//!
//! // The in-memory exporter needs no collector. Without the `otel` feature the
//! // handle is an inactive no-op seam, so the call still succeeds.
//! let config = OtelTracerConfig::default().with_exporter_type(ExporterType::Memory);
//! let handle = init_otel_tracer(config).expect("the memory exporter always initializes");
//! assert_eq!(handle.is_active(), cfg!(feature = "otel"));
//! assert_eq!(handle.tracer_name(), "openjiuwen.tracer.otel");
//! handle.force_flush();
//! handle.shutdown();
//! handle.shutdown(); // idempotent
//! ```
//!
//! # Tracer integration
//!
//! ```
//! # #[cfg(feature = "otel")]
//! # {
//! use std::sync::Arc;
//!
//! use openjiuwen_core::session::tracer::{InvokeType, TraceSink, Tracer};
//! use openjiuwen_extensions::otel::{
//!     init_otel_tracer, ExporterType, OtelTraceSink, OtelTracerConfig,
//! };
//!
//! let handle = init_otel_tracer(
//!     OtelTracerConfig::default().with_exporter_type(ExporterType::Memory),
//! )
//! .expect("the memory exporter initializes");
//! let sink: Arc<dyn TraceSink> = OtelTraceSink::new(&handle).expect("a live handle builds a sink");
//! let tracer = Tracer::with_sink("session-1", "trace-1", sink);
//!
//! let span = tracer.start_span("Model", InvokeType::Llm);
//! span.set_inputs(serde_json::json!({"messages": [{"role": "user", "content": "hi"}]}));
//! span.finish(serde_json::json!({"content": "hello"}));
//! handle.force_flush();
//!
//! // The session tracer emits one terminal frame per span; the sink opens the
//! // OTel span from that frame, so a finished span is still exported.
//! let exporter = handle.capture().expect("the memory exporter is attached");
//! assert_eq!(exporter.names(), vec!["chat Model".to_owned()]);
//! # }
//! ```
//!
//! [`TraceSink`]: openjiuwen_core::session::tracer::TraceSink
//! [`TraceSpan`]: openjiuwen_core::session::tracer::TraceSpan

pub mod config;
pub mod handle;
pub mod redaction;
pub mod semconv;
pub mod span_names;

#[cfg(feature = "otel")]
mod agent_handler;
#[cfg(feature = "otel")]
mod attrs;
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

pub use config::{global_instrument_enabled, ExporterType, OtelTracerConfig, OtlpProtocol};
pub use handle::{init_otel_tracer, TracerHandle};
pub use redaction::{hash_value, truncate, RedactionField, Redactor};
pub use span_names::{
    chat_span_name, chat_span_name_for_model, embeddings_span_name, is_llm_component,
    is_retrieval_component, is_tool_component, legacy_tool_span_name, simple_span_name,
    tool_span_name, workflow_component_span_name,
};

#[cfg(feature = "otel")]
pub use agent_handler::OtelAgentHandler;
#[cfg(feature = "otel")]
pub use handle::init_otel_tracer_with_exporter;
#[cfg(feature = "otel")]
pub use provider::{tracing_layer, CaptureExporter};
#[cfg(feature = "otel")]
pub use sink::OtelTraceSink;
#[cfg(feature = "otel")]
pub use span_manager::{OtelAgentSpanManager, OtelSpanState, OtelWorkflowSpanManager};
#[cfg(feature = "otel")]
pub use workflow_handler::OtelWorkflowHandler;
