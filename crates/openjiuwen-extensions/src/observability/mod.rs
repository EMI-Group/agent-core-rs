//! Dependency-neutral observability utilities.
//!
//! Port of `openjiuwen/extensions/observability/` (v0.1.20): the parts of that package that are
//! pure data, string and attribute plumbing. Python's package is importable without initializing
//! an exporter (semantic conventions, redaction and content addressing are plain helpers), and
//! this module keeps the same split: everything here compiles into the default build except the
//! two modules that hand a `SpanData` batch to the SDK.
//!
//! # What lives here
//!
//! - [`project_semconv`] — the project-owned attribute namespaces (`agentteam.*`, `deepagent.*`
//!   and the `openjiuwen.*` keys [`crate::otel::semconv`] does not own), the trajectory
//!   data-plane key sets and versions, and the pinned upstream `GenAI` registry metadata.
//! - [`redaction`] — credential masking (`***`), error summary and stack-trace redaction.
//! - [`content_addressing`] — `@oj-seq` prefix-chain content addressing for restated `gen_ai.*`
//!   payload attributes.
//! - [`tool_outcome`] — tool-failure classification (`ToolReportedFailure`) over a tool result.
//! - [`error_reporting`] — the `error_reason` priority chain behind an error span's message.
//! - `file_exporter` (`otel` feature) — the collector-free OTLP-JSON WAL behind
//!   [`ExporterType::File`](crate::otel::ExporterType::File).
//! - [`deferred`] — the typed `Unsupported` seams for the SDK-only modules below.
//! - `transforming` (`otel` feature) — export-time span projection, so a backend view never
//!   consumes the collection-time attribute budget.
//!
//! # Coverage against the Python package
//!
//! | Python module | Rust counterpart |
//! | --- | --- |
//! | `semconv.py`, `gen_ai_semconv.py` | [`project_semconv`] plus [`crate::otel::semconv`] |
//! | `redaction.py` | [`redaction`] (error/credential half) and [`crate::otel::redaction`] (prompt/completion half) |
//! | `content_addressing.py`, the id half of `otlp_codec.py` | [`content_addressing`]; the envelope encoding is `crate::otel::otlp` |
//! | `tool_outcome.py`, the pure half of `error_reporting.py` | [`tool_outcome`], [`error_reporting`] |
//! | `file_exporter.py` | `file_exporter` |
//! | `exporters/transforming.py` | `transforming` |
//! | `exporters/langfuse.py` | [`crate::otel::langfuse`] |
//! | `config.py` | [`crate::otel::OtelTracerConfig`] (including `traces_dir` / `file_retention_days`) |
//! | `callback_handler.py`, `span_context.py` | `crate::otel::OtelAgentHandler`, `crate::otel::OtelWorkflowHandler`, `crate::otel::OtelAgentSpanManager` |
//!
//! # What is deferred
//!
//! The remaining Python modules describe a *process-global* SDK runtime or a research data plane,
//! both of which this crate deliberately does not carry.
//! Each deferred capability keeps a typed entry point in [`deferred`] that answers
//! [`Error::Unsupported`](openjiuwen_common::error::Error::Unsupported) with the reason and the
//! supported alternative, so a Python-parity caller never meets a missing symbol:
//!
//! - `setup.py`, `demand.py`, `runtime.py` → [`deferred::init_observability`] and the demand
//!   acquire/release seam. One process-wide `TracerProvider` with reference-counted acquire is not
//!   carried: this crate keeps provider lifecycle in [`crate::otel::TracerHandle`], one handle per
//!   ownership scope, and never installs a global exporter or spins up a background runtime on the
//!   caller's thread.
//! - `span_record_processor.py`, `trajectory_events.py`, `span_context.py`,
//!   `context_compression_handler.py` → [`deferred::publish_stream_frame`] and the trajectory
//!   commit/consumer seams. They need `ReadableSpan`-shaped frozen spans plus a reader-side
//!   consumer registry, which this workspace has no trajectory data plane for; the pure parts
//!   (record field sets, key literals, `@oj-seq` addressing) are ported in [`content_addressing`]
//!   and [`project_semconv`].
//! - `instrumentation.py` → [`deferred::ensure_global_http_instrumentation`]. Monkey-patching
//!   `httpx`/`requests`/`aiohttp`/`FastAPI` has no Rust equivalent hook surface; a consumer
//!   instruments its own transport.
//!
//! [`deferred::is_initialized`] is the one infallible answer (`false`) and
//! [`deferred::instrumentation_env_flag`] the one portable constant, both documented as such.
//! No deferred path is a silent no-op: nothing is initialized, exported, patched or replaced.

pub mod content_addressing;
pub mod deferred;
pub mod error_reporting;
pub mod project_semconv;
pub mod redaction;
pub mod tool_outcome;

#[cfg(feature = "otel")]
pub mod file_exporter;

#[cfg(feature = "otel")]
pub mod transforming;

pub use content_addressing::{
    address_payload, addressable_attributes, build_sequence, content_hash,
    parse_sequence_reference, rebuild_value, sequence_reference, split_elements, AddressedSequence,
    SequenceNode, ADDRESSED_ATTRIBUTE_KEYS, SEQUENCE_REFERENCE_PREFIX, SEQUENCE_REFERENCE_VERSION,
};
pub use deferred::{
    acquire_observability_demand, emit_compaction_window_commit, emit_context_window_commit,
    ensure_global_http_instrumentation, force_flush_provider, init_observability,
    instrumentation_env_flag, is_initialized, publish_span_snapshot, publish_stream_frame,
    register_span_consumer, release_observability_demand,
    reset_global_http_instrumentation_for_tests, shutdown_observability, tracer,
    ObservabilityDemand, INSTRUMENTATION_DEFERRED_REASON, PROVIDER_LIFECYCLE_DEFERRED_REASON,
    TRAJECTORY_DATA_PLANE_DEFERRED_REASON,
};
pub use error_reporting::error_reason;
pub use redaction::{
    mask_bearer_tokens, mask_secrets, redact_error_stacktrace, redact_error_summary, BEARER_MASK,
    ERROR_SUMMARY_MAX_LENGTH, MASK, REDACTED_HASH_PREFIX,
};
pub use tool_outcome::{
    tool_failure_reason, tool_result_for_exception, DEFAULT_FAILURE_REASON,
    EXCEPTION_RESULT_PREFIX, TOOL_REPORTED_FAILURE,
};

#[cfg(feature = "otel")]
pub use file_exporter::TraceFileExporter;

#[cfg(feature = "otel")]
pub use transforming::{clone_span_with_attributes, SpanTransform, TransformingSpanExporter};
