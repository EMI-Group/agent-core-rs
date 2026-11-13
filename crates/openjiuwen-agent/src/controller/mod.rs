//! Controller stack: routing events to agents and workflows.
//!
//! Ports `openjiuwen/core/controller/`: an [`Event`] carries user input or a
//! task signal, an [`IntentRecognizer`] classifies it, a [`Planner`] turns the
//! intents into [`Task`]s, and a [`TaskScheduler`] dispatches each task to the
//! [`TaskExecutor`] registered for its task type (a [`ControllerAgent`], a
//! [`WorkflowController`], or a host-supplied executor).
//!
//! The turn loop lives in [`BaseController`]: it owns the bounded event queue,
//! the recognizer, the optional [`TaskRouter`] (the LLM executor selector), the
//! planner, and the executor registry, and it persists its task state in the
//! session under [`TASKS_STATE_KEY`] so a later turn resumes with the same
//! tasks. A paused task is an `input-required` status on the output, never an
//! `Err`.
//!
//! # Modules
//!
//! - [`schema`] — the data model: payload frames, events, intents, tasks,
//!   statuses, and the batch/streaming controller output.
//! - [`config`] — [`ControllerConfig`] and [`DefaultResponse`].
//! - [`intent`] — [`IntentRecognizer`] (rule-based [`RuleIntentRecognizer`],
//!   model-backed [`LlmIntentRecognizer`]) and the [`Planner`].
//! - [`executors`] — the executor seam, the registry, the sequential
//!   [`TaskScheduler`], and the three built-in executors.
//! - [`base`] — [`Controller`]/[`BaseController`], the [`EventHandler`] seam,
//!   and the persisted task state.
//! - [`legacy`] — Python's pre-event-queue lineage: legacy value objects, the
//!   [`IntentDetector`], and the string-routed [`IntentDetectionController`].
//!
//! # Deviations from Python
//!
//! - The new pipeline and the legacy lineage share one crate but no types;
//!   `legacy::schema::LegacyTask` is not `schema::Task`.
//! - Python's background `TaskManager`/`TaskScheduler` loop is a single-turn,
//!   in-process dispatch here: tasks run sequentially in intent order and every
//!   failure becomes a failed task rather than an error.
//! - The LLM intent recognizer replaces Python's tool-call loop with a
//!   tolerant JSON-array contract.
pub mod base;
pub mod config;
pub mod executors;
pub mod intent;
pub mod legacy;
pub mod schema;
pub use base::{BaseController, Controller, EventHandler, EventHandlerOutcome, TASKS_STATE_KEY};
pub use config::{ControllerConfig, DefaultResponse};
pub use executors::{
    ControllerAgent, ExecutorInfo, LLMController, TaskContext, TaskExecutor, TaskExecutorRegistry,
    TaskRouter, TaskRun, TaskScheduler, WorkflowController, AGENT_TASK_TYPE, LLM_TASK_TYPE,
    WORKFLOW_TASK_TYPE,
};
pub use intent::{
    IntentRecognizer, IntentRule, LlmIntentRecognizer, Planner, RuleIntentRecognizer,
    LLM_INTENT_PROMPT,
};
pub use legacy::{IntentDetectionController, IntentDetector, LegacyPlanner};
pub use schema::{
    frames_text, ControllerOutput, ControllerOutputChunk, ControllerOutputPayload, DataFrame,
    Event, EventType, FileDataFrame, Intent, IntentType, JsonDataFrame, OutputKind, Task,
    TaskResult, TaskStatus, TextDataFrame, ALL_TASKS_PROCESSED, CONTROLLER_OUTPUT_TYPE,
    TASK_PROCESSING,
};
