//! Dependency-neutral observability utilities.
//!
//! Port of `openjiuwen/extensions/observability/` (v0.1.20): the parts of that package that are
//! pure data, string and attribute plumbing, with the OpenTelemetry SDK left to [`crate::otel`].
//! Python's package is deliberately importable without initializing an exporter — semantic
//! conventions, redaction and content addressing are plain helpers — and this module keeps the
//! same split: everything here compiles into the default build except the two modules that hand a
//! `SpanData` batch to the SDK.
//!
//! # What lives here
//!
//! - [`project_semconv`] — the project-owned attribute namespaces (`agentteam.*`, `deepagent.*`),
//!   the trajectory data-plane key sets and versions, and the pinned upstream GenAI revision.
//! - [`redaction`] — credential masking (`***`), error summary/stack-trace redaction, and the
//!   error-reason priority chain that fills `error.type` and `openjiuwen.error`.
//! - [`content_addressing`] — `@oj-seq` prefix-chain content addressing for restated `gen_ai.*`
//!   payload attributes.
//! - [`tool_outcome`] — tool-failure classification (`ToolReportedFailure`) over a tool result.
//! - `file_exporter` (`otel` feature) — the collector-free OTLP-JSON WAL
//!   (`ExporterType::File`).
//! - `transforming` (`otel` feature) — export-time span projection, so a backend view never
//!   consumes the collection-time attribute budget.
//!
//! # What is deferred
//!
//! The SDK adapter layer stays in `openjiuwen/extensions/observability/` in Python and has no Rust
//! counterpart here: `callback_handler.py` (the LLM/tool/agent span lifecycle over the Python
//! callback framework — this workspace's equivalent is [`crate::otel::OtelAgentHandler`] plus
//! [`crate::otel::OtelWorkflowHandler`]), `span_context.py`'s contextvar span tracker,
//! `span_record_processor.py`'s async fan-out, `trajectory_events.py`'s event channel,
//! `instrumentation.py`'s monkey-patched HTTP instrumentors, `context_compression_handler.py` and
//! `demand.py`/`setup.py`'s process-global provider lifecycle. Their Python shapes depend on
//! SDK-only types (`ReadableSpan`, contextvars, module patching) or on a single process-global
//! provider; this crate keeps provider lifecycle in [`crate::otel::TracerHandle`], one handle per
//! ownership scope, and never installs a global exporter.

pub mod content_addressing;
pub mod error_reporting;
pub mod project_semconv;
pub mod redaction;
pub mod tool_outcome;

#[cfg(feature = "otel")]
pub mod file_exporter;

#[cfg(feature = "otel")]
pub mod transforming;
