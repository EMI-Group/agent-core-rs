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
//!
//! # Deviation from Python
//!
//! Python splits the goal capability into a session-state store, a `GoalManager` that owns every
//! write under an interaction control lock, and a per-round `GoalReportSink` consumed by
//! `TaskCompletionRail`. Only the data half is ported here: a file-backed [`GoalStore`] with one
//! async lock per instance, and tools that commit the model's assessment directly. The
//! loop-side behaviour — attempt scheduling, pause/resume/overwrite policy, event emission and
//! the `goal.updated` interaction events — lives in the goal rail (`crate::rails::goal`), which
//! owns the `GoalManager` role and drives these types through [`GoalStore::load`],
//! [`GoalStore::update`] and [`GoalRecord::progress_summary`].
//!
//! Two smaller differences: the persisted record is one `goal.json` instead of session state
//! (see [`GoalStore::for_workspace`]), and an unknown persisted status degrades to `active`
//! instead of clearing the goal (see [`GoalRecord::from_dict`]).
mod record;
mod schema;
mod store;
#[cfg(test)]
mod tests;
mod tools;

pub use record::GoalRecord;
pub use schema::{
    now_iso, parse_timestamp, GoalAssessment, GoalAssessmentStatus, GoalStatus, TokenUsage,
    DEFAULT_SESSION_ID,
};
pub use store::GoalStore;
pub use tools::{
    GetCurrentGoalArgs, GetCurrentGoalTool, SubmitGoalReportArgs, SubmitGoalReportTool,
};
