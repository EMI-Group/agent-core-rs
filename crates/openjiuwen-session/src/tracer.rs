//! Trace positions recorded with checkpoints.
//!
//! The full tracer (spans, frames, sinks) is introduced by a later phase; this
//! module currently exposes only [`TracePosition`], the resume point stored in
//! [`crate::checkpoint::Checkpoint`]. The type keeps Python's `trace_state`
//! shape (`trace_id`, `last_invoke_id`, `frame_count`).
use serde::{Deserialize, Serialize};

/// The position of the last emitted trace frame within a session.
///
/// Checkpoints persist the position so a resumed run can continue the trace
/// where it stopped instead of starting a new one.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TracePosition {
    /// The trace identifier shared by every frame of the run.
    pub trace_id: String,
    /// The invoke id of the most recently emitted span, if any.
    pub last_invoke_id: Option<String>,
    /// The number of frames emitted so far.
    pub frame_count: u64,
}
