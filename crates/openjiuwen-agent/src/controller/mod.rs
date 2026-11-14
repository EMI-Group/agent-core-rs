//! Controller stack: routing events to agents and workflows.
//!
//! Ports `openjiuwen/core/controller/`: an [`Event`] carries user input or a
//! task signal, an [`IntentRecognizer`] classifies it, a [`Planner`] turns the
//! intents into [`Task`]s in the shared [`TaskManager`], and the controller
//! dispatches each task to the [`TaskExecutor`] registered for its task type (a
//! [`ControllerAgent`], a [`WorkflowController`], or a host-supplied executor).
//!
//! The turn loop lives in [`BaseController`]: it owns the bounded event queue,
//! the recognizer, the optional [`TaskRouter`] (the LLM executor selector), the
//! planner, the [`TaskManager`], the [`Reasoner`], and the executor registry,
//! and it persists its task state in the session under [`TASKS_STATE_KEY`] so a
//! later turn resumes with the same tasks. One [`Controller::invoke`] drains the
//! events queued when it starts, recognizes and plans, dispatches its batch of
//! tasks synchronously, and aggregates one [`ControllerOutput`]; a paused task is
//! an `input-required` status on that output, never an `Err`.
//!
//! Three seams extend the single turn:
//!
//! - [`TaskSchedulerLoop`] runs the background half: it scans the [`TaskManager`]
//!   for submitted tasks, admits them under
//!   [`ControllerConfig::max_concurrent_tasks`], drives pause/cancel through the
//!   [`TaskControl`] seam, and reports completion. [`BaseController::start`]
//!   launches it, so work added to the manager outside a turn (by a
//!   [`Reasoner`], a host, or a restored session) still runs.
//! - [`Reasoner`] is the task-operation state machine: [`Reasoner::plan_ops`]
//!   maps one [`Intent`] to [`TaskOp`]s, [`Reasoner::reason`] records a
//!   clarification, and [`Reasoner::apply_task_ops`] applies the ops to the
//!   manager, returning one [`TaskOpOutcome`] per op.
//! - [`GroupRouter`] is the agent-group seam behind
//!   [`BaseController::send_to_agent`] and [`BaseController::publish`]: the
//!   router owns point-to-point sends and subscription publishes, and a
//!   controller whose agent belongs to no group fails with the typed
//!   configuration error of [`group::send_without_group`] /
//!   [`group::publish_without_group`].
//!
//! # Modules
//!
//! - [`schema`] — the data model: payload frames, events, intents, tasks,
//!   statuses, and the batch/streaming controller output.
//! - [`config`] — [`ControllerConfig`] and [`DefaultResponse`].
//! - [`intent`] — [`IntentRecognizer`] (rule-based [`RuleIntentRecognizer`],
//!   model-backed [`LlmIntentRecognizer`]) and the [`Planner`].
//! - [`task_manager`] — the shared, indexed task store ([`TaskManager`],
//!   [`TaskFilter`]) and its per-session snapshot ([`TaskManagerState`]).
//! - [`reasoner`] — the intent-to-[`TaskOp`] planner and applier plus the
//!   [`TaskControl`] seam.
//! - [`executors`] — the executor seam, the registry, the batch
//!   [`TaskScheduler`], the background [`TaskSchedulerLoop`], and the three
//!   built-in executors.
//! - [`base`] — [`Controller`]/[`BaseController`], the [`EventHandler`] seam,
//!   and the persisted task state.
//! - [`queue`] — the per-conversation [`TaskQueue`] of [`RunningTask`]s.
//! - [`group`] — the [`GroupRouter`] seam and its typed errors.
//! - [`legacy`] — Python's pre-event-queue lineage: legacy value objects, the
//!   [`IntentDetector`], and an [`IntentDetectionController`] wrapper that adds
//!   per-conversation preemption, group routing, and its own task queue on top
//!   of [`BaseController`].
//!
//! # Deviations from Python
//!
//! - The new pipeline and the legacy lineage share one crate but no types;
//!   `legacy::schema::LegacyTask` is not `schema::Task`.
//! - There is no asyncio event queue or pub/sub: the bounded FIFO is drained by
//!   the turn that enqueued into it, and a dispatched task's result is recorded
//!   on the task itself instead of being streamed to a queue.
//! - Admission and ordering are deterministic: the scheduler scans in priority
//!   descending order with `created_seq` and `task_id` as tiebreaks, where
//!   Python iterates an unordered selection.
//! - `ControllerConfig::max_concurrent_tasks == 0` means unlimited here;
//!   Python's `>=` comparison against `0` blocks every admission.
//! - `metadata["depends_on"]` dependency gating is a Rust extension: a
//!   [`TaskStatus::Waiting`] task whose dependencies are all completed is
//!   promoted to `submitted` at the top of a scheduler scan.
//! - Synchronous turn dispatch is kept: [`Controller::invoke`] returns the batch
//!   its events produced, and the background loop is an additive path for tasks
//!   that no turn dispatches.
//! - The LLM intent recognizer replaces Python's tool-call loop with a
//!   tolerant JSON-array contract.
pub mod base;
pub mod config;
pub mod executors;
pub mod group;
pub mod intent;
pub mod legacy;
pub mod queue;
pub mod reasoner;
pub mod schema;
pub mod task_manager;
pub use base::{BaseController, Controller, EventHandler, EventHandlerOutcome, TASKS_STATE_KEY};
pub use config::{ControllerConfig, DefaultResponse};
pub use executors::{
    ControllerAgent, ExecutorInfo, LLMController, SessionProvider, TaskContext, TaskExecutor,
    TaskExecutorRegistry, TaskRouter, TaskRun, TaskScheduler, TaskSchedulerLoop,
    WorkflowController, AGENT_TASK_TYPE, LLM_TASK_TYPE, WORKFLOW_TASK_TYPE,
};
pub use group::{GroupMethod, GroupRouter};
pub use intent::{
    IntentRecognizer, IntentRule, LlmIntentRecognizer, Planner, RuleIntentRecognizer,
    LLM_INTENT_PROMPT,
};
pub use legacy::{IntentDetectionController, IntentDetector, LegacyPlanner};
pub use queue::{RunningTask, TaskQueue};
pub use reasoner::{Reasoner, TaskControl, TaskOp, TaskOpOutcome};
pub use schema::{
    frames_text, ControllerOutput, ControllerOutputChunk, ControllerOutputPayload, DataFrame,
    Event, EventType, FileDataFrame, Intent, IntentType, JsonDataFrame, OutputKind, Task,
    TaskResult, TaskStatus, TextDataFrame, ALL_TASKS_PROCESSED, CONTROLLER_OUTPUT_TYPE,
    TASK_PROCESSING,
};
pub use task_manager::{
    PrioritySelector, TaskFilter, TaskIdSelector, TaskManager, TaskManagerState,
};

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::RwLock;
    use std::sync::Arc;

    #[test]
    fn task_manager_and_filter_default_to_an_empty_store() {
        let manager = TaskManager::new();
        assert!(manager.is_empty());
        assert_eq!(manager.len(), 0);

        let filter = TaskFilter::new();
        assert!(
            filter.validate().is_err(),
            "an empty filter names no selector"
        );
        let by_status = TaskFilter::by_status(TaskStatus::Submitted);
        assert!(by_status.validate().is_ok());
        assert_eq!(TaskIdSelector::One("t1".to_owned()).to_vec(), ["t1"]);
        assert_eq!(PrioritySelector::Level(3).level(), Some(3));
        assert!(PrioritySelector::Highest.is_highest());
        let state = TaskManagerState::new(Vec::new(), 0);
        assert!(state.is_empty());
        assert_eq!(state.task_ids(), Vec::<String>::new());
    }

    #[test]
    fn task_queue_and_group_router_seams_are_reachable() {
        let queue = TaskQueue::new();
        assert!(queue.is_empty());
        assert!(!queue.has_running("c1"));

        assert_eq!(GroupMethod::Publish.method_name(), "publish");
        assert_eq!(
            GroupMethod::SendToAgent("a1").to_string(),
            "send_to_agent('a1')"
        );
        let missing: Option<&Arc<dyn GroupRouter>> = None;
        let Err(error) = group::require_group(missing, "ctl", GroupMethod::Publish) else {
            panic!("no router is installed");
        };
        assert!(error.message().ends_with(group::NOT_IN_GROUP));
    }

    #[test]
    fn reasoner_defaults_and_task_ops_are_reachable() {
        let reasoner = Reasoner::new(ControllerConfig::new(), None);
        assert!(reasoner.control().is_none());
        assert_eq!(
            reasoner.config().default_task_type,
            ControllerConfig::new().default_task_type
        );
        let pause = TaskOp::Pause {
            task_id: "t1".to_owned(),
        };
        assert_eq!(pause.kind(), "pause");
        assert_eq!(pause.task_id(), Some("t1"));
        assert_eq!(
            TaskOp::Unknown {
                prompt: "which task?".to_owned()
            }
            .task_id(),
            None
        );
        let outcome = TaskOpOutcome {
            kind: "pause",
            task_id: Some("t1".to_owned()),
            status: None,
            applied: false,
            message: "no task control installed".to_owned(),
            clarification: None,
        };
        assert!(!outcome.applied);
    }

    #[test]
    fn scheduler_loop_builds_over_an_empty_registry() {
        let manager = TaskManager::new();
        let registry = Arc::new(RwLock::new(TaskExecutorRegistry::new()));
        let config = Arc::new(RwLock::new(ControllerConfig::new()));
        let provider: SessionProvider = Arc::new(|_session_id: &str| None);
        let scheduler = TaskSchedulerLoop::new(manager.clone(), registry, config, "ctl", provider);
        assert!(!scheduler.is_running());
        assert!(scheduler.manager().is_empty());
    }
}
