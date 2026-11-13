//! Success detection over a recorded [`Trajectory`](crate::trajectory::Trajectory).
//!
//! Ports the offline intent of Python `openjiuwen.agent_evolving.ttse.success`:
//! a run counts as successful when its recorded outcome says so, or when the
//! outcome is undecided but the run shows enough tool activity and no error
//! step. The detector is pure and deterministic — no model, no clock.

/// Decides whether one trajectory counts as a successful experience.
pub struct SuccessDetector;
