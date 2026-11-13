//! Goal tracking tools and the persisted goal record.
//!
//! Ports `openjiuwen/harness/goal/{schema,store,manager}.py` and the model-facing half of
//! `openjiuwen/harness/tools/goal.py` (`get_current_goal`, `submit_goal_report`).
//!
//! # Required public API (registry contract)
//!
//! - `GoalStatus`, `GoalAssessmentStatus`, `GoalAssessment`, `GoalRecord` — serde data types
//!   with the Python field names and snake_case JSON.
//! - `GoalStore::new(path: PathBuf) -> Arc<Self>`, `GoalStore::for_workspace(&Workspace) ->
//!   Arc<Self>`, plus load/save/start/assess helpers used by the tools and the rail.
//! - `GetCurrentGoalTool::new(store: Arc<GoalStore>, language: impl Into<String>) -> Self` and
//!   `SubmitGoalReportTool::new(store: Arc<GoalStore>, language: impl Into<String>) -> Self`,
//!   each with `card_for(language: &str) -> ToolCard`; model-facing names `get_current_goal`
//!   and `submit_goal_report`.
