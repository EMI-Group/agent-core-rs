//! Runner-level human-in-the-loop (HITL) broker for active runs.
//!
//! Registers the runs the runner is executing, delivers inbound human payloads
//! to a run's inbox, answers pending interaction requests on a run's session,
//! and pauses/resumes/cancels an active run.
