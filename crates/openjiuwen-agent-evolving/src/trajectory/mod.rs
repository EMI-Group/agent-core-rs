//! Trajectories: recording, grouping and windowing an agent run.
//!
//! Ports the portable part of Python `openjiuwen.agent_evolving.trajectory`:
//! the trajectory schema (`trajectory/model.py`, `trajectory/schema.py`), span
//! bookkeeping (`trajectory/spans.py`), windowing (`trajectory/windows.py`) and
//! the offline builder/extractor (`trajectory/offline/`).
//!
//! - [`Trajectory`] / [`TrajectoryStep`] — the serde record of one run.
//! - [`TrajectoryBuilder`] / [`from_messages`] — incremental construction from
//!   messages and steps.
//! - [`TrajectoryStore`] — in-memory grouping by session/case, with
//!   deterministic ordering.
//! - [`spans`] — tool-call/result pairing with open-span markers.
//! - [`windows`] / [`render_digest`] / [`Trajectory::digest`] — bounded views
//!   for prompt injection.
//!
//! Everything is offline: no clocks in recorded values, no model calls.

mod builder;
mod model;
mod spans;
mod store;
mod windows;

pub use builder::{from_messages, TrajectoryBuilder};
pub use model::{
    StepKind, StepStatus, Trajectory, TrajectoryDigest, TrajectoryOutcome, TrajectoryStep,
};
pub use spans::{failed_spans, longest_span, span_by_name, spans, TrajectorySpan};
pub use store::{TrajectoryGroupSummary, TrajectoryStats, TrajectoryStore};
pub use windows::{first_error, last_n, render_digest, window_around, windows, TrajectoryWindow};
