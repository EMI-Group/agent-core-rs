//! Optional integrations and backend adapters for the openJiuwen Rust SDK.
//!
//! Maps to the Python package `openjiuwen.extensions` (v0.1.20). Nothing in the
//! kernel depends on this crate, and every external service or heavy dependency
//! sits behind a Cargo feature (see [`crate`] docs in `CONTEXT.md`).
//!
//! Feature summary:
//!
//! - `a2a` — the Agent-to-Agent protocol client and server ([`a2a`]).
//! - `otel` — live OpenTelemetry tracing; without it [`otel`] is a no-op seam.
//! - `sqlite` / `redis` / `pulsar` — optional backends for the adapters.
//!
//! ```
//! use openjiuwen_extensions::otel::{init_otel_tracer, ExporterType, OtelTracerConfig};
//!
//! // The in-memory exporter needs no collector. Without the `otel` feature the
//! // handle is an inactive no-op seam, so the call still succeeds.
//! let config = OtelTracerConfig::default().with_exporter_type(ExporterType::Memory);
//! let handle = init_otel_tracer(config).expect("the memory exporter always initializes");
//! assert_eq!(handle.is_active(), cfg!(feature = "otel"));
//! ```
#![forbid(unsafe_code)]

pub mod checkpointer;
pub mod context_evolver;
pub mod message_queue;
pub mod otel;
pub mod store;
pub mod sys_operation;
pub mod vendor_specific;

#[cfg(feature = "a2a")]
pub mod a2a;

/// Returns the crate name.
#[must_use]
pub fn crate_name() -> &'static str {
    "openjiuwen-extensions"
}
