//! Goal tracking: the persisted goal record, its store and the two model-facing goal tools.
//!
//! Ports `openjiuwen/harness/goal/{schema,store,manager}.py` and the model-facing half of
//! `openjiuwen/harness/tools/goal.py` (`get_current_goal`, `submit_goal_report`).
//!
//! # API
//!
//! - [`GoalStatus`], [`GoalAssessmentStatus`], [`GoalAssessment`], [`TokenUsage`] and
//!   [`GoalRecord`] — serde data types with the Python field names and `snake_case` JSON keys.
//! - [`GoalStore`] — `new(path)`, `for_session(path, session_id)`, `for_workspace(&Workspace)`,
//!   `path()`, `session_id()`, `lock()`, `load()`, `save(&record)`, `start(objective, now)`,
//!   `update(|record| …, now)` and `clear()`; every mutation receives `now`.
//! - [`GetCurrentGoalTool`] and [`SubmitGoalReportTool`] — `new(store, language)` (plus
//!   `unavailable(language)` when no store is wired) and `card_for(language)`; model-facing
//!   names `get_current_goal` and `submit_goal_report`.
//! - [`GoalRail`] and [`GoalRailConfig`] — the loop-side counterpart: before every outer task
//!   iteration it publishes the goal reminder prompt section while the goal is active, nudges
//!   the model to verify progress every `heartbeat_interval` iterations, and appends a
//!   `goal.updated` event under [`GOAL_EVENT_EXTRA_KEY`] once the goal settles.
//!
//! # Goal rail
//!
//! The rail is built with `GoalRail::new(Arc<GoalStore>, GoalRailConfig)` and registered on a
//! `HarnessRailRegistry`. `DeepAgentBuilder::build` and `resolve_deep_agent_parts` register it
//! automatically when a mounted tool is `get_current_goal` or `submit_goal_report`, and a host
//! may also register it directly. It publishes the prompt section named `goal` (title
//! `Current Goal`) and queues its interaction events in `ctx.extra` under
//! [`GOAL_EVENT_EXTRA_KEY`] (`harness.goal_events`), because rails have no direct event sink;
//! the host drains that JSON array after the task-iteration hooks and forwards each entry to the
//! interaction stream, since the `DeepAgent` loop exposes no drain seam. A corrupt goal file is
//! logged and skipped, never fatal.
//!
//! `crate::tools::goal::GoalRail` is the only path: `crate::rails` deliberately does not
//! re-export it, because `rails` must not depend on `tools`. This module is the one place in
//! `crate::tools` that touches `crate::rails`.
//!
//! # Deviation from Python
//!
//! Python splits the goal capability into a session-state store, a `GoalManager` that owns every
//! write under an interaction control lock, and a per-round `GoalReportSink` consumed by
//! `TaskCompletionRail`. Only the data half is ported here: a file-backed [`GoalStore`] with one
//! async lock per instance, and tools that commit the model's assessment directly. The
//! loop-side behaviour — attempt scheduling, pause/resume/overwrite policy, event emission and
//! the `goal.updated` interaction events — lives in the goal rail
//! ([`GoalRail`]), which owns the `GoalManager` role and drives these types through
//! [`GoalStore::load`], [`GoalStore::update`] and [`GoalRecord::progress_summary`].
//!
//! Two smaller differences: the persisted record is one `goal.json` instead of session state
//! (see [`GoalStore::for_workspace`]), and an unknown persisted status degrades to `active`
//! instead of clearing the goal (see [`GoalRecord::from_dict`]).
mod rail;
mod record;
mod schema;
mod store;
#[cfg(test)]
mod tests;
mod tools;

pub use rail::{GoalRail, GoalRailConfig, GOAL_EVENT_EXTRA_KEY};
pub use record::GoalRecord;
pub use schema::{
    now_iso, parse_timestamp, GoalAssessment, GoalAssessmentStatus, GoalStatus, TokenUsage,
    DEFAULT_SESSION_ID,
};
pub use store::GoalStore;
pub use tools::{
    GetCurrentGoalArgs, GetCurrentGoalTool, SubmitGoalReportArgs, SubmitGoalReportTool,
};
