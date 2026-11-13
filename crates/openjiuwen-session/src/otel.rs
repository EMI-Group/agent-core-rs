//! Feature-gated trace-sink installation for a session.
//!
//! This crate owns no `OTel` dependency: the real sink lives in
//! `openjiuwen-extensions` (`otel` feature) and is handed in as an
//! `Arc<dyn TraceSink>`. [`install_trace_sink`] is the single place product and
//! adapter layers call so a session can be traced by a real exporter without
//! knowing about the backend.
//!
//! # Feature behavior
//!
//! - With the `otel` feature enabled the helper forwards to
//!   [`crate::Session::install_trace_sink`], so the session (and every span it
//!   starts afterwards) routes to the supplied sink.
//! - Without the feature it is a **deliberate, documented no-op** that logs at
//!   debug level and returns `Ok(())`: a build that never links an `OTel`
//!   exporter must not fail a run that merely asks for one, and it must never
//!   return [`openjiuwen_common::error::ErrorKind::Unsupported`]. Callers that
//!   need to distinguish "installed" from "ignored" must configure it through
//!   [`crate::Session::install_trace_sink`] directly.
use std::sync::Arc;

use openjiuwen_common::error::Result;

use crate::session::Session;
use crate::tracer::TraceSink;

/// Installs `sink` as the trace sink of `session`.
///
/// With the `otel` feature this forwards to the session hook; without it, it
/// logs at debug level and returns `Ok(())` without touching the session (see
/// the module docs: this no-op is intentional and never an error).
///
/// # Errors
///
/// With the `otel` feature, propagates the session's own error (for example
/// [`openjiuwen_common::error::ErrorKind::Unsupported`] from a read-only
/// wrapper session). Without the feature it never fails.
#[cfg(feature = "otel")]
pub fn install_trace_sink(session: &dyn Session, sink: Arc<dyn TraceSink>) -> Result<()> {
    session.install_trace_sink(sink)
}

/// Installs `sink` as the trace sink of `session`.
///
/// With the `otel` feature this forwards to the session hook; without it, it
/// logs at debug level and returns `Ok(())` without touching the session (see
/// the module docs: this no-op is intentional and never an error).
///
/// # Errors
///
/// With the `otel` feature, propagates the session's own error (for example
/// [`openjiuwen_common::error::ErrorKind::Unsupported`] from a read-only
/// wrapper session). Without the feature it never fails.
// The by-value `Arc` keeps this signature identical to the feature-enabled
// variant, so callers compile against either build.
#[cfg(not(feature = "otel"))]
#[allow(clippy::needless_pass_by_value)]
pub fn install_trace_sink(_session: &dyn Session, _sink: Arc<dyn TraceSink>) -> Result<()> {
    tracing::debug!(
        target: "openjiuwen_session::otel",
        "otel feature is disabled; trace sink installation is a no-op"
    );
    Ok(())
}
