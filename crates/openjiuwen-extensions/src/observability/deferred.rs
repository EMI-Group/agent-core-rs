//! Typed deferral seam for the observability capabilities this crate does not carry.
//!
//! Python's `openjiuwen/extensions/observability/` (v0.1.20) also ships a process-global SDK
//! runtime, process-wide HTTP instrumentation and a trajectory data plane.
//! Rather than leaving those symbols absent, this module keeps their Python-shaped entry points
//! and answers [`Error::Unsupported`] with a reason naming both the missing capability and the
//! supported alternative, so a Python-parity consumer learns the boundary at the call site instead
//! of from a missing symbol or a silent no-op.
//!
//! | Python module | Why it cannot be ported | Supported alternative in this crate |
//! | --- | --- | --- |
//! | `setup.py`, `demand.py`, `runtime.py` | one process-wide `TracerProvider` with reference-counted acquire/release demands and a dedicated background runtime; OpenTelemetry allows exactly one global provider per process, and a hidden runtime would block the caller's thread and mutate process-global state | `crate::otel::init_otel_tracer` returns a `TracerHandle` per ownership scope, whose `force_flush`/`shutdown_with_timeout` cover the same lifecycle; `crate::otel::OtelTraceSink` and `crate::otel::tracing_layer` are the sink and layer seams |
//! | `instrumentation.py` | monkey-patches `httpx`, `requests`, `aiohttp` and `fastapi`; Rust has no module-patching hook surface for patching another library's transport process-wide | a consumer instruments its own transport and sets its own `traceparent` header; the `OPENJIUWEN_OTEL_GLOBAL_INSTRUMENT_ENABLE` switch itself is portable and resolved by `crate::otel::global_instrument_enabled` |
//! | `span_record_processor.py`, `trajectory_events.py`, `span_context.py`, `context_compression_handler.py` | needs frozen `ReadableSpan` records, a reader-side consumer registry and a contextvar span tracker that advances a subject's canonical context window — i.e. a trajectory data plane | the data shapes are ported in [`crate::observability::project_semconv`] (record kinds and key literals) and [`crate::observability::content_addressing`] (`@oj-seq` chains); `crate::otel::OtelTraceSink` is the span seam a consumer drives |
//!
//! # What stays portable, and why
//!
//! [`instrumentation_env_flag`] returns Python's `ENV_FLAG` literal and [`ObservabilityDemand`] is
//! Python's closed `{"agent", "team"}` key set: both are data, not capabilities, so neither is an
//! error.
//! [`is_initialized`] answers `false` for the same reason Python's `is_initialized()` returns a
//! `bool`: this crate never installs a process-global provider, so the honest answer is invariant,
//! and an error would only push callers into an `unwrap_or(false)` that hides it.
//!
//! # Error contract
//!
//! Every fallible entry point returns
//! [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported) carrying one of
//! [`PROVIDER_LIFECYCLE_DEFERRED_REASON`], [`INSTRUMENTATION_DEFERRED_REASON`] or
//! [`TRAJECTORY_DATA_PLANE_DEFERRED_REASON`] plus a shared tail; the only other error is
//! [`ObservabilityDemand`]'s `FromStr`, which rejects an unknown key with
//! [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation) like Python's
//! `ValueError`.

use std::fmt;
use std::str::FromStr;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

use crate::otel::OtelTracerConfig;

/// `tracing` target of every event this module emits.
const TRACING_TARGET: &str = "openjiuwen_extensions::observability::deferred";

/// Reason carried by the process-global provider lifecycle entry points.
///
/// Names the Python runtime, what is missing (one global provider with reference-counted demands
/// and a background runtime) and the alternative (a per-scope `TracerHandle` plus the sink/layer
/// seams).
pub const PROVIDER_LIFECYCLE_DEFERRED_REASON: &str = "the process-global observability runtime (Python `setup.py`, `demand.py` and `runtime.py`) is deferred: it keeps exactly one process-wide OpenTelemetry `TracerProvider`, reference-counted acquire/release demands and a dedicated background runtime, and this crate installs neither a global provider nor a runtime on the caller's thread; use `crate::otel::init_otel_tracer`, which returns a `TracerHandle` per ownership scope whose `force_flush` and `shutdown_with_timeout` cover the same lifecycle, and drive spans through `crate::otel::OtelTraceSink` or `crate::otel::tracing_layer`";

/// Reason carried by the process-wide HTTP instrumentation entry points.
///
/// Names the Python module, what is missing (a module-patching hook surface) and the alternative
/// (instrument your own transport; the switch is still resolved by `global_instrument_enabled`).
pub const INSTRUMENTATION_DEFERRED_REASON: &str = "process-wide HTTP instrumentation (Python `instrumentation.py`) is deferred: Python monkey-patches `httpx`, `requests`, `aiohttp` and `fastapi` so every outbound request carries a `traceparent` header, and Rust has no module-patching hook surface for patching another library's transport process-wide; instrument your own transport instead (open a span from your own tracer and set the `traceparent` header in your HTTP client), and resolve Python's `OPENJIUWEN_OTEL_GLOBAL_INSTRUMENT_ENABLE` switch with `crate::otel::global_instrument_enabled`, which is supported";

/// Reason carried by the trajectory data-plane entry points.
///
/// Names the Python modules, what is missing (frozen `ReadableSpan` records, a consumer registry
/// and the contextvar span tracker) and the alternative (the ported data shapes plus
/// `OtelTraceSink`).
pub const TRAJECTORY_DATA_PLANE_DEFERRED_REASON: &str = "the trajectory data plane (Python `span_record_processor.py`, `trajectory_events.py`, `span_context.py` and `context_compression_handler.py`) is deferred: it fans frozen `ReadableSpan` records, span snapshots and model-stream frames out to a reader-side consumer registry and keeps a contextvar span tracker that advances a subject's canonical context window, none of which this crate carries; the data shapes it needs are ported in `crate::observability::project_semconv` (trajectory record kinds and key literals) and `crate::observability::content_addressing` (`@oj-seq` chains), and `crate::otel::OtelTraceSink` is the span seam a consumer can drive from its own pipeline";

/// Tail shared by every deferral message in this module.
const DEFERRED_TAIL: &str =
    " (typed deferral: nothing was initialized, exported, patched or replaced)";

/// Python's `instrumentation.ENV_FLAG` literal.
const INSTRUMENTATION_ENV_FLAG: &str = "OPENJIUWEN_OTEL_GLOBAL_INSTRUMENT_ENABLE";

/// The closed set of observability runtimes that may hold a provider demand.
///
/// Python's `demand.py` accepts exactly `{"agent", "team"}` on purpose: an unknown key would create
/// a demand nobody releases.
/// The wire spellings are the lowercase keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ObservabilityDemand {
    /// The single-agent runtime (Python key `"agent"`).
    Agent,
    /// The multi-agent team runtime (Python key `"team"`).
    Team,
}

impl ObservabilityDemand {
    /// Every demand key, in Python's declaration order.
    pub const ALL: [Self; 2] = [Self::Agent, Self::Team];

    /// The wire spelling of this demand (`"agent"` or `"team"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Agent => "agent",
            Self::Team => "team",
        }
    }
}

impl fmt::Display for ObservabilityDemand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for ObservabilityDemand {
    type Err = Error;

    /// Parses a Python runtime key into a demand.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when `value` is not one of the two closed keys, naming the accepted
    /// spellings (the counterpart of Python's `ValueError`).
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "agent" => Ok(Self::Agent),
            "team" => Ok(Self::Team),
            other => Err(Error::validation(format!(
                "unknown observability runtime: {other}; expected one of `agent`, `team`"
            ))),
        }
    }
}

/// Initializes the process-global observability runtime — deferred.
///
/// Python's `setup.init_observability(config, *, span_exporter_override,
/// additional_span_processors)` installs one global provider; this port only logs `config` and
/// changes nothing.
///
/// # Errors
///
/// [`Error::Unsupported`] always; use `crate::otel::init_otel_tracer(config)` instead.
pub fn init_observability(config: &OtelTracerConfig) -> Result<()> {
    tracing::debug!(
        target: TRACING_TARGET,
        service_name = %config.service_name,
        exporter = ?config.exporter_type,
        "process-global observability initialization is deferred"
    );
    Err(deferred(PROVIDER_LIFECYCLE_DEFERRED_REASON))
}

/// Shuts down the process-global observability runtime — deferred.
///
/// Python's `setup.shutdown_observability()`; there is no global provider to shut down here.
///
/// # Errors
///
/// [`Error::Unsupported`] always; drop the `TracerHandle` from `crate::otel::init_otel_tracer`
/// instead (its `shutdown_with_timeout` covers the lifecycle).
pub fn shutdown_observability() -> Result<()> {
    Err(deferred(PROVIDER_LIFECYCLE_DEFERRED_REASON))
}

/// Force-flushes the process-global observability provider — deferred.
///
/// Python's `setup.force_flush_provider(timeout_millis=5000)` has no target here.
///
/// # Errors
///
/// [`Error::Unsupported`] always; call `TracerHandle::force_flush` on your own handle.
pub fn force_flush_provider() -> Result<()> {
    Err(deferred(PROVIDER_LIFECYCLE_DEFERRED_REASON))
}

/// Reports whether the process-global observability runtime is initialized.
///
/// Always `false`, and deliberately not an error: Python's `setup.is_initialized()` returns a
/// `bool`, and this crate installs no process-global provider.
/// Use `TracerHandle::is_active` to ask whether a handle of your own scope is live.
#[must_use]
pub const fn is_initialized() -> bool {
    false
}

/// Returns a tracer bound to the process-global runtime — deferred.
///
/// Python's `setup.get_tracer(name)` resolves the tracer from the global provider; a live tracer
/// cannot exist without one, so the `()` payload is only a placeholder a caller can hold.
///
/// # Errors
///
/// [`Error::Unsupported`] always; build your tracer from the provider behind
/// `crate::otel::init_otel_tracer` instead.
pub fn tracer(name: &str) -> Result<()> {
    tracing::debug!(
        target: TRACING_TARGET,
        name,
        "global-runtime tracer lookup is deferred"
    );
    Err(deferred(PROVIDER_LIFECYCLE_DEFERRED_REASON))
}

/// Acquires one runtime's demand for the shared provider — deferred.
///
/// Python's `demand.acquire_observability_demand(runtime, *, observability_config, initializer)`
/// reference-counts the one global provider; a per-scope `TracerHandle` needs no bookkeeping, so
/// `runtime` is only logged.
///
/// # Errors
///
/// [`Error::Unsupported`] always; hold your scope's `TracerHandle` while tracing is needed.
pub fn acquire_observability_demand(runtime: ObservabilityDemand) -> Result<()> {
    tracing::debug!(
        target: TRACING_TARGET,
        runtime = %runtime,
        "observability demand acquisition is deferred"
    );
    Err(deferred(PROVIDER_LIFECYCLE_DEFERRED_REASON))
}

/// Releases one runtime's demand for the shared provider — deferred.
///
/// Python's `demand.release_observability_demand(runtime, *, finalizer)` shuts the provider down
/// once the last demand is gone.
///
/// # Errors
///
/// [`Error::Unsupported`] always; dropping that scope's `TracerHandle` is the supported release.
pub fn release_observability_demand(runtime: ObservabilityDemand) -> Result<()> {
    tracing::debug!(
        target: TRACING_TARGET,
        runtime = %runtime,
        "observability demand release is deferred"
    );
    Err(deferred(PROVIDER_LIFECYCLE_DEFERRED_REASON))
}

/// Returns Python's `instrumentation.ENV_FLAG` literal.
///
/// The literal is data, not a capability, so it is not an error; the live resolver
/// [`crate::otel::global_instrument_enabled`] reads it over
/// [`OtelTracerConfig::global_instrument_enable`].
#[must_use]
pub const fn instrumentation_env_flag() -> &'static str {
    INSTRUMENTATION_ENV_FLAG
}

/// Instruments the HTTP libraries once per process — deferred.
///
/// Python's `instrumentation.ensure_global_http_instrumentation(provider_factory)` returns `true`
/// once the HTTP libraries are patched; the `bool` stays for shape parity only.
///
/// # Errors
///
/// [`Error::Unsupported`] always, carrying [`INSTRUMENTATION_DEFERRED_REASON`]; instrument your own
/// transport instead.
pub fn ensure_global_http_instrumentation() -> Result<bool> {
    Err(deferred(INSTRUMENTATION_DEFERRED_REASON))
}

/// Resets the once-per-process instrumentation latch — deferred.
///
/// Python's `instrumentation.reset_global_http_instrumentation_for_tests()` is test-only; there is
/// no latch and no patched class here, and a test must learn that no such state exists.
///
/// # Errors
///
/// [`Error::Unsupported`] always, carrying [`INSTRUMENTATION_DEFERRED_REASON`].
pub fn reset_global_http_instrumentation_for_tests() -> Result<()> {
    Err(deferred(INSTRUMENTATION_DEFERRED_REASON))
}

/// Registers a consumer of ended-span records — deferred.
///
/// Python's `SpanRecordProcessor.register_consumer(consumer)` fans every ended span out to a
/// registered consumer; `consumer_name` is only logged here.
///
/// # Errors
///
/// [`Error::Unsupported`] always, carrying [`TRAJECTORY_DATA_PLANE_DEFERRED_REASON`].
pub fn register_span_consumer(consumer_name: &str) -> Result<()> {
    tracing::debug!(
        target: TRACING_TARGET,
        consumer_name,
        "span consumer registration is deferred"
    );
    Err(deferred(TRAJECTORY_DATA_PLANE_DEFERRED_REASON))
}

/// Publishes one ended-span snapshot to the registered consumers — deferred.
///
/// Python's `demand.publish_span_snapshot(span, update_kind)` is best-effort; `update_kind` and the
/// payload size are logged, never the payload.
///
/// # Errors
///
/// [`Error::Unsupported`] always, carrying [`TRAJECTORY_DATA_PLANE_DEFERRED_REASON`].
pub fn publish_span_snapshot(update_kind: &str, snapshot: &serde_json::Value) -> Result<()> {
    tracing::debug!(
        target: TRACING_TARGET,
        update_kind,
        payload_bytes = payload_bytes(snapshot),
        "span snapshot publication is deferred"
    );
    Err(deferred(TRAJECTORY_DATA_PLANE_DEFERRED_REASON))
}

/// Publishes one model-stream frame to the frame-capable consumers — deferred.
///
/// Python's `demand.publish_stream_frame(record)` is best-effort; only the payload size is logged.
///
/// # Errors
///
/// [`Error::Unsupported`] always, carrying [`TRAJECTORY_DATA_PLANE_DEFERRED_REASON`].
pub fn publish_stream_frame(frame: &serde_json::Value) -> Result<()> {
    tracing::debug!(
        target: TRACING_TARGET,
        payload_bytes = payload_bytes(frame),
        "stream frame publication is deferred"
    );
    Err(deferred(TRAJECTORY_DATA_PLANE_DEFERRED_REASON))
}

/// Emits the `context.window.commit` span of a request that carries the conversation forward —
/// deferred.
///
/// Python's `trajectory_events.emit_context_window_commit(*, tracer, llm_span, messages,
/// request_purpose)` returns the emitted span (`None` for a `"compaction"` purpose); only
/// `request_purpose` and the payload size are logged.
///
/// # Errors
///
/// [`Error::Unsupported`] always, carrying [`TRAJECTORY_DATA_PLANE_DEFERRED_REASON`].
pub fn emit_context_window_commit(
    request_purpose: &str,
    messages: &serde_json::Value,
) -> Result<()> {
    tracing::debug!(
        target: TRACING_TARGET,
        request_purpose,
        payload_bytes = payload_bytes(messages),
        "context window commit emission is deferred"
    );
    Err(deferred(TRAJECTORY_DATA_PLANE_DEFERRED_REASON))
}

/// Emits the context window a completed compaction produced — deferred.
///
/// Python's `trajectory_events.emit_compaction_window_commit(*, tracer, parent_span, messages,
/// operation_id, model_requests)` states the rewritten window itself; only `operation_id` and the
/// payload size are logged.
///
/// # Errors
///
/// [`Error::Unsupported`] always, carrying [`TRAJECTORY_DATA_PLANE_DEFERRED_REASON`].
pub fn emit_compaction_window_commit(
    operation_id: &str,
    messages: &serde_json::Value,
) -> Result<()> {
    tracing::debug!(
        target: TRACING_TARGET,
        operation_id,
        payload_bytes = payload_bytes(messages),
        "compaction window commit emission is deferred"
    );
    Err(deferred(TRAJECTORY_DATA_PLANE_DEFERRED_REASON))
}

/// Builds the deferral error and records it once at `debug`.
///
/// `reason` names the missing capability and the supported alternative; [`DEFERRED_TAIL`] states
/// that nothing changed, so no message can read as a silent no-op.
fn deferred(reason: &'static str) -> Error {
    tracing::debug!(
        target: TRACING_TARGET,
        reason,
        "observability capability is deferred"
    );
    Error::unsupported(format!("{reason}{DEFERRED_TAIL}"))
}

/// Size of `payload` as compact JSON, for the deferral log only.
///
/// The payload is measured and discarded: its contents are never logged.
fn payload_bytes(payload: &serde_json::Value) -> usize {
    serde_json::to_string(payload).map_or(0, |rendered| rendered.len())
}
