//! Trajectories: recording, grouping and windowing an agent run.
//!
//! Ports the portable part of Python `openjiuwen.agent_evolving.trajectory`:
//! the trajectory schema (`trajectory/model.py`, `trajectory/schema.py`), span
//! bookkeeping (`trajectory/spans.py`), windowing (`trajectory/windows.py`) and
//! the offline builder/extractor (`trajectory/offline/`).
//!
//! - [`Trajectory`] / [`TrajectoryStep`] — the serde record of one run.
//! - [`TrajectoryBuilder`] — incremental construction from messages and steps.
//! - [`TrajectoryStore`] — in-memory grouping by session/team/case, with
//!   deterministic ordering.
//! - Window and digest helpers — bounded views for prompt injection.
//!
//! Everything is offline: no clocks in recorded values, no model calls.
