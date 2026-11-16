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
//! - With the `otel` feature enabled [`try_install_trace_sink`] forwards to
//!   [`crate::Session::install_trace_sink`] and reports
//!   [`TraceSinkInstall::Installed`], so the session (and every span it starts
//!   afterwards) routes to the supplied sink.
//! - Without the feature it is an **explicit, observable no-op**: it logs at
//!   warn level naming the disabled `otel` feature and returns
//!   [`TraceSinkInstall::FeatureDisabled`]. A build that never links an `OTel`
//!   exporter must not fail a run that merely asks for one, and it must never
//!   return [`openjiuwen_common::error::ErrorKind::Unsupported`] — but it must
//!   also not pretend the sink was installed. Callers that need to act on the
//!   difference inspect the outcome of [`try_install_trace_sink`] (or
//!   [`is_enabled`]) instead of guessing.
//! - [`install_trace_sink`] is the compatibility wrapper: same signature in both
//!   builds, discards the outcome, and therefore keeps its historical
//!   always-`Ok(())` shape in a feature-off build.
use std::sync::Arc;

use openjiuwen_common::error::Result;
use serde::{Deserialize, Serialize};

use crate::session::Session;
use crate::tracer::TraceSink;

/// What [`try_install_trace_sink`] did.
///
/// `Copy`, `snake_case` on the wire (`"installed"` / `"feature_disabled"`), and
/// not `#[must_use]`, so a caller may ignore it as easily as it can inspect it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceSinkInstall {
    /// The session accepted the sink.
    Installed,
    /// The crate was built without the `otel` feature, so nothing was installed.
    FeatureDisabled,
}

impl TraceSinkInstall {
    /// Whether a sink was actually installed.
    #[must_use]
    pub const fn is_installed(&self) -> bool {
        matches!(self, Self::Installed)
    }
}

/// Whether this build can install a trace sink (`cfg!(feature = "otel")`).
///
/// A `const fn` so a caller can branch at compile time as well as run time.
#[must_use]
pub const fn is_enabled() -> bool {
    cfg!(feature = "otel")
}

/// Installs `sink` as the trace sink of `session`, reporting what happened.
///
/// With the `otel` feature this forwards to the session hook and returns
/// [`TraceSinkInstall::Installed`]; without it, it logs a warning naming the
/// `otel` feature and returns [`TraceSinkInstall::FeatureDisabled`] without
/// touching the session.
///
/// # Errors
///
/// With the `otel` feature, propagates the session's own error (for example
/// [`openjiuwen_common::error::ErrorKind::Unsupported`] from a read-only
/// wrapper session). Without the feature it never fails.
#[cfg(feature = "otel")]
pub fn try_install_trace_sink(
    session: &dyn Session,
    sink: Arc<dyn TraceSink>,
) -> Result<TraceSinkInstall> {
    session.install_trace_sink(sink)?;
    Ok(TraceSinkInstall::Installed)
}

/// Installs `sink` as the trace sink of `session`, reporting what happened.
///
/// With the `otel` feature this forwards to the session hook and returns
/// [`TraceSinkInstall::Installed`]; without it, it logs a warning naming the
/// `otel` feature and returns [`TraceSinkInstall::FeatureDisabled`] without
/// touching the session.
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
pub fn try_install_trace_sink(
    _session: &dyn Session,
    _sink: Arc<dyn TraceSink>,
) -> Result<TraceSinkInstall> {
    tracing::warn!(
        target: "openjiuwen_session::otel",
        "otel feature is disabled; the trace sink was NOT installed \
         (enable openjiuwen-session/otel to observe spans)"
    );
    Ok(TraceSinkInstall::FeatureDisabled)
}

/// Installs `sink` as the trace sink of `session`.
///
/// Compatibility wrapper over [`try_install_trace_sink`] that discards the
/// outcome: with the `otel` feature it installs (and propagates the session's
/// error), and without it, it is the documented no-op that always returns
/// `Ok(())`. Use [`try_install_trace_sink`] to observe which happened.
///
/// # Errors
///
/// With the `otel` feature, propagates the session's own error (for example
/// [`openjiuwen_common::error::ErrorKind::Unsupported`] from a read-only
/// wrapper session). Without the feature it never fails.
pub fn install_trace_sink(session: &dyn Session, sink: Arc<dyn TraceSink>) -> Result<()> {
    try_install_trace_sink(session, sink).map(|_outcome| ())
}
