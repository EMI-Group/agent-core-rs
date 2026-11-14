//! Task-operation reasoner: intents in, task operations out.
//!
//! Python keeps this reasoning path inside `EventHandlerWithIntentRecognition`
//! (`openjiuwen/core/controller/modules/intent_recognizer.py`): every recognized
//! [`Intent`] is dispatched to one `_process_*_intent` coroutine that creates,
//! pauses, resumes, continues, supplements, cancels, or modifies a [`Task`], and
//! an unrecognized intent is answered with a clarification prompt.
//! (`legacy/reasoner/agent_reasoner.py` is an empty artifact.) This module is
//! that state machine, split into a pure planning step and an applying step so a
//! host can inspect or pre-empt the operations:
//!
//! 1. [`Reasoner::plan_ops`] maps one intent to zero or more [`TaskOp`]s without
//!    touching any state. A confidence below
//!    [`ControllerConfig::intent_confidence_threshold`] degrades the intent to
//!    exactly one [`TaskOp::Unknown`].
//! 2. [`Reasoner::reason`] adds the clarification side effect of an unknown
//!    intent: the prompt is stored in the session under
//!    [`CLARIFICATION_STATE_KEY`] and mirrored as one `task_interaction` frame.
//! 3. [`Reasoner::apply_task_ops`] executes ops against the shared
//!    [`TaskManager`] and the [`Planner`], returning one [`TaskOpOutcome`] per
//!    op in input order. Task-level interruption goes through the optional
//!    [`TaskControl`] seam (the scheduler's pause/cancel/resume); without one, a
//!    control op is reported as not applied instead of failing.
//!
//! Planning is deterministic: no clock, no randomness, no model call. Task ids
//! come from the [`Planner`] (`<session>_task_<seq>`), so two creates never
//! collide even across turns. Applying never aborts a batch — a duplicate id, a
//! missing task, or a refused control call becomes a failed [`TaskOpOutcome`],
//! never an `Err` and never a panic.
//!
//! # Deviations from Python
//!
//! - Python handles seven intent kinds in `handle_input` and lets everything
//!   else fall into `_process_unknown_task_intent`; here
//!   [`IntentType::SwitchTask`] is a first-class operation that cancels the
//!   current task and then creates the new one, matching the intent's documented
//!   meaning.
//! - Python only writes a `{"clarification_prompt": …}` dict to the session
//!   stream; this port also records the prompt, the originating event id, and
//!   the confidence under [`CLARIFICATION_STATE_KEY`] so a later turn can
//!   correlate the answer.
//! - [`TaskOp::Continue`] cannot pull context-engine messages (this layer owns no
//!   context engine), so it copies the dependency's recorded inputs, context id,
//!   and task type instead.
//! - [`TaskOp::Modify`] creates a replacement task linked to the original rather
//!   than mutating the original in place, so the audit trail of the old task is
//!   preserved.
//! - A blank target id, blank supplement/modify payload, or empty dependency list
//!   is treated as an unrecognized intent instead of raising Python's
//!   `build_error`, because the reasoner reports outcomes rather than errors.
/// Session state key holding the pending clarification of an unknown intent.
pub const CLARIFICATION_STATE_KEY: &str = "__controller_clarification__";
/// Clarification asked when an unknown intent carries no prompt.
pub const DEFAULT_CLARIFICATION: &str = "Please clarify your request.";
/// Marker appended to a task description by [`TaskOp::Supplement`].
pub const SUPPLEMENT_MARKER: &str = "\n\n任务补充信息:\n";
/// Marker appended to a task description by [`TaskOp::Modify`].
pub const MODIFY_MARKER: &str = "\n\n修改详情:\n";
use crate::controller::config::ControllerConfig;
use crate::controller::intent::Planner;
use crate::controller::schema::{
    ControllerOutputChunk, ControllerOutputPayload, IntentType, OutputKind,
};
pub use crate::controller::schema::{Event, Intent, Task, TaskStatus};
use crate::controller::task_manager::TaskManager;
use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_session::Session;
use parking_lot::RwLock;
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::Arc;
use tracing::warn;
/// Kind string of [`TaskOp::Create`].
const KIND_CREATE: &str = "create";
/// Kind string of [`TaskOp::Pause`].
const KIND_PAUSE: &str = "pause";
/// Kind string of [`TaskOp::Resume`].
const KIND_RESUME: &str = "resume";
/// Kind string of [`TaskOp::Cancel`].
const KIND_CANCEL: &str = "cancel";
/// Kind string of [`TaskOp::Continue`].
const KIND_CONTINUE: &str = "continue";
/// Kind string of [`TaskOp::Supplement`].
const KIND_SUPPLEMENT: &str = "supplement";
/// Kind string of [`TaskOp::Modify`].
const KIND_MODIFY: &str = "modify";
/// Kind string of [`TaskOp::Unknown`].
const KIND_UNKNOWN: &str = "unknown";
/// One task operation derived from an intent.
///
/// A plan is a flat list of these: most intents yield one op,
/// [`IntentType::SwitchTask`] yields a [`TaskOp::Cancel`] followed by a
/// [`TaskOp::Create`], and a gated or unrecognized intent yields exactly one
/// [`TaskOp::Unknown`].
#[derive(Debug, Clone, PartialEq)]
pub enum TaskOp {
    /// Create and schedule a new task for `intent`, scheduled as `task_type`.
    Create {
        /// The intent the task is planned from.
        intent: Intent,
        /// Task type the new task is dispatched to.
        task_type: String,
    },
    /// Pause a working task.
    Pause {
        /// Id of the targeted task.
        task_id: String,
    },
    /// Resume a paused task.
    Resume {
        /// Id of the targeted task.
        task_id: String,
    },
    /// Cancel a working task.
    Cancel {
        /// Id of the targeted task.
        task_id: String,
    },
    /// Continue from one or more completed tasks.
    Continue {
        /// The upstream tasks the new task builds on.
        depend_task_id: Vec<String>,
        /// The event that triggered the continuation.
        event: Event,
        /// Metadata the new task carries.
        metadata: BTreeMap<String, serde_json::Value>,
        /// Description of the new task; falls back to the dependency's.
        description: Option<String>,
    },
    /// Append information to a task that waits for it and re-schedule it.
    Supplement {
        /// Id of the targeted task.
        task_id: String,
        /// The supplied information.
        info: String,
    },
    /// Replace a task with a modified version of itself.
    Modify {
        /// Id of the targeted task.
        task_id: String,
        /// Details of the requested modification.
        details: String,
    },
    /// Recognition needs a clarification before anything can run.
    Unknown {
        /// Question asked to the caller.
        prompt: String,
    },
}
impl TaskOp {
    /// Returns the operation kind (`"create"`, `"pause"`, `"resume"`,
    /// `"cancel"`, `"continue"`, `"supplement"`, `"modify"`, `"unknown"`).
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Create { .. } => KIND_CREATE,
            Self::Pause { .. } => KIND_PAUSE,
            Self::Resume { .. } => KIND_RESUME,
            Self::Cancel { .. } => KIND_CANCEL,
            Self::Continue { .. } => KIND_CONTINUE,
            Self::Supplement { .. } => KIND_SUPPLEMENT,
            Self::Modify { .. } => KIND_MODIFY,
            Self::Unknown { .. } => KIND_UNKNOWN,
        }
    }
    /// Returns the targeted task id, when the op names one.
    ///
    /// [`TaskOp::Create`] and [`TaskOp::Unknown`] return `None` because the task
    /// does not exist yet (the planner mints the id), and
    /// [`TaskOp::Continue`] reports its first dependency.
    #[must_use]
    pub fn task_id(&self) -> Option<&str> {
        match self {
            Self::Create { .. } | Self::Unknown { .. } => None,
            Self::Pause { task_id }
            | Self::Resume { task_id }
            | Self::Cancel { task_id }
            | Self::Supplement { task_id, .. }
            | Self::Modify { task_id, .. } => Some(task_id),
            Self::Continue { depend_task_id, .. } => depend_task_id.first().map(String::as_str),
        }
    }
}
/// What applying one [`TaskOp`] did.
#[derive(Debug, Clone, PartialEq)]
pub struct TaskOpOutcome {
    /// Kind of the op that produced the outcome ([`TaskOp::kind`]).
    pub kind: &'static str,
    /// Task the outcome refers to, when it refers to one.
    pub task_id: Option<String>,
    /// Status the targeted task left with, when the op changed it.
    pub status: Option<TaskStatus>,
    /// Whether the operation took effect.
    pub applied: bool,
    /// Human-readable result, including the failure reason when not applied.
    pub message: String,
    /// Clarification the caller must answer, for [`TaskOp::Unknown`].
    pub clarification: Option<String>,
}
/// The task-level control seam: pause, cancel, and resume a scheduled task.
///
/// The controller loop installs the scheduler here; a host without a scheduler
/// leaves it unset, which makes control ops report
/// `"no task control installed"` instead of failing.
#[async_trait]
pub trait TaskControl: Send + Sync {
    /// Pauses the task; returns whether it was paused.
    async fn pause(&self, task_id: &str) -> bool;
    /// Cancels the task; returns whether it was cancelled.
    async fn cancel(&self, task_id: &str) -> bool;
    /// Resumes the task; returns whether it was resumed.
    async fn resume(&self, task_id: &str) -> bool;
}
/// Turns intents into task operations and applies them.
///
/// The reasoner is a cheap handle (`Arc` inside) shared by the controller loop
/// and its helpers. It owns the [`ControllerConfig`] the confidence gate,
/// default task type, and default priority are read from, plus the optional
/// [`TaskControl`] seam; per-invocation state (the [`TaskManager`] and the
/// [`Planner`]) is passed in, so one reasoner serves every session.
#[derive(Clone, Default)]
pub struct Reasoner {
    /// The controller configuration the gate and defaults come from.
    config: Arc<RwLock<ControllerConfig>>,
    /// The task-control seam, when the host installed one.
    control: Arc<RwLock<Option<Arc<dyn TaskControl>>>>,
}
impl Reasoner {
    /// Builds a reasoner over `config` with an optional control seam.
    #[must_use]
    pub fn new(config: ControllerConfig, control: Option<Arc<dyn TaskControl>>) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            control: Arc::new(RwLock::new(control)),
        }
    }
    /// Returns a copy of the current configuration.
    #[must_use]
    pub fn config(&self) -> ControllerConfig {
        self.config.read().clone()
    }
    /// Replaces the configuration.
    pub fn set_config(&self, config: ControllerConfig) {
        *self.config.write() = config;
    }
    /// Returns the installed control seam, if any.
    #[must_use]
    pub fn control(&self) -> Option<Arc<dyn TaskControl>> {
        self.control.read().clone()
    }
    /// Installs or clears the control seam.
    pub fn set_control(&self, control: Option<Arc<dyn TaskControl>>) {
        *self.control.write() = control;
    }
    /// Maps one intent to the task operations it asks for.
    ///
    /// A confidence strictly below
    /// [`ControllerConfig::intent_confidence_threshold`] (or a `NaN` one)
    /// degrades the intent to exactly one [`TaskOp::Unknown`] with the intent's
    /// clarification prompt. Otherwise the intent kind selects the operations:
    ///
    /// - [`IntentType::CreateTask`] → [`TaskOp::Create`] with the intent's own
    ///   task type or [`ControllerConfig::default_task_type`].
    /// - [`IntentType::PauseTask`], [`IntentType::ResumeTask`], and
    ///   [`IntentType::CancelTask`] → the matching op, or [`TaskOp::Unknown`]
    ///   when the target id is missing or blank.
    /// - [`IntentType::ContinueTask`] → [`TaskOp::Continue`] over the non-blank
    ///   dependency ids, or [`TaskOp::Unknown`] when none is left.
    /// - [`IntentType::SupplementTask`] and [`IntentType::ModifyTask`] → the
    ///   matching op, or [`TaskOp::Unknown`] when the payload is blank.
    /// - [`IntentType::SwitchTask`] → [`TaskOp::Cancel`] of the current task
    ///   (only when it names one) followed by a [`TaskOp::Create`] of the new
    ///   task, whose intent is retyped as [`IntentType::CreateTask`].
    /// - [`IntentType::UnknownTask`] → [`TaskOp::Unknown`].
    #[must_use]
    pub fn plan_ops(&self, intent: &Intent) -> Vec<TaskOp> {
        let config = self.config();
        let gated =
            intent.confidence.is_nan() || intent.confidence < config.intent_confidence_threshold;
        if gated {
            return vec![unknown_op(intent)];
        }
        let task_type = intent
            .task_type
            .clone()
            .unwrap_or_else(|| config.default_task_type.clone());
        match intent.intent_type {
            IntentType::CreateTask => vec![TaskOp::Create {
                intent: intent.clone(),
                task_type,
            }],
            IntentType::PauseTask => match target_id(intent) {
                Some(task_id) => vec![TaskOp::Pause {
                    task_id: task_id.to_owned(),
                }],
                None => vec![unknown_op(intent)],
            },
            IntentType::ResumeTask => match target_id(intent) {
                Some(task_id) => vec![TaskOp::Resume {
                    task_id: task_id.to_owned(),
                }],
                None => vec![unknown_op(intent)],
            },
            IntentType::CancelTask => match target_id(intent) {
                Some(task_id) => vec![TaskOp::Cancel {
                    task_id: task_id.to_owned(),
                }],
                None => vec![unknown_op(intent)],
            },
            IntentType::ContinueTask => {
                let depend_task_id: Vec<String> = intent
                    .depend_task_id
                    .iter()
                    .map(|id| id.trim())
                    .filter(|id| !id.is_empty())
                    .map(str::to_owned)
                    .collect();
                if depend_task_id.is_empty() {
                    return vec![unknown_op(intent)];
                }
                vec![TaskOp::Continue {
                    depend_task_id,
                    event: intent.event.clone(),
                    metadata: intent.metadata.clone(),
                    description: optional_text(intent.target_task_description.as_deref()),
                }]
            }
            IntentType::SupplementTask => {
                let Some(task_id) = target_id(intent) else {
                    return vec![unknown_op(intent)];
                };
                match optional_text(intent.supplementary_info.as_deref()) {
                    Some(info) => vec![TaskOp::Supplement {
                        task_id: task_id.to_owned(),
                        info,
                    }],
                    None => vec![unknown_op(intent)],
                }
            }
            IntentType::ModifyTask => {
                let Some(task_id) = target_id(intent) else {
                    return vec![unknown_op(intent)];
                };
                match optional_text(intent.modification_details.as_deref()) {
                    Some(details) => vec![TaskOp::Modify {
                        task_id: task_id.to_owned(),
                        details,
                    }],
                    None => vec![unknown_op(intent)],
                }
            }
            IntentType::SwitchTask => {
                let mut ops = Vec::with_capacity(2);
                if let Some(task_id) = target_id(intent) {
                    ops.push(TaskOp::Cancel {
                        task_id: task_id.to_owned(),
                    });
                }
                ops.push(TaskOp::Create {
                    intent: switch_create_intent(intent),
                    task_type,
                });
                ops
            }
            IntentType::UnknownTask => vec![unknown_op(intent)],
        }
    }
    /// Plans the operations of `intent` and asks the caller to clarify when the
    /// plan is a single [`TaskOp::Unknown`].
    ///
    /// The clarification is persisted on `session` under
    /// [`CLARIFICATION_STATE_KEY`] as `{"prompt", "event_id", "confidence"}` and
    /// mirrored as one final `task_interaction`
    /// [`ControllerOutputChunk`]. Session failures are logged and never abort
    /// the reasoning step.
    ///
    /// # Errors
    ///
    /// Currently infallible: a blank target id, a blank supplement or modify
    /// payload, and an empty dependency list all degrade to
    /// [`TaskOp::Unknown`] rather than an error, and the session writes are
    /// best-effort. The [`Result`] keeps a validation stage available without
    /// changing the call sites.
    pub async fn reason(
        &self,
        intent: &Intent,
        session: Option<&Arc<dyn Session>>,
    ) -> Result<Vec<TaskOp>> {
        let ops = self.plan_ops(intent);
        let Some(TaskOp::Unknown { prompt }) = ops.first() else {
            return Ok(ops);
        };
        let Some(session) = session else {
            return Ok(ops);
        };
        let state = json!({
            CLARIFICATION_STATE_KEY: {
                "prompt": prompt,
                "event_id": intent.event.event_id,
                "confidence": intent.confidence,
            },
        });
        if let Err(error) = session.update_state(state).await {
            warn!(%error, "cannot record the controller clarification state");
        }
        let chunk = ControllerOutputChunk::last(
            0,
            ControllerOutputPayload::text(OutputKind::TaskInteraction, prompt.clone()),
        );
        if let Err(error) = session.write_stream(chunk.to_stream_item()).await {
            warn!(%error, "cannot publish the controller clarification frame");
        }
        Ok(ops)
    }
    /// Applies `ops` in order and returns one outcome per op.
    ///
    /// The batch never aborts: every op is attempted, and a missing task, a
    /// duplicate id, or a refused control call is reported as an unapplied
    /// [`TaskOpOutcome`]. Operations that schedule a task do so through the
    /// planner, so ids stay unique across turns.
    ///
    /// `session_id` scopes the planned task ids and context identities; when it
    /// is blank the id of `session` is used instead, and `session` is otherwise
    /// unused.
    pub async fn apply_task_ops(
        &self,
        ops: &[TaskOp],
        manager: &TaskManager,
        planner: &mut Planner,
        session_id: &str,
        session: Option<&Arc<dyn Session>>,
    ) -> Vec<TaskOpOutcome> {
        let config = self.config();
        let scope = resolve_scope(session_id, session);
        let mut ctx = OpContext {
            manager,
            planner,
            session_id: &scope,
            config: &config,
        };
        let mut outcomes = Vec::with_capacity(ops.len());
        for op in ops {
            outcomes.push(self.apply_op(&mut ctx, op).await);
        }
        outcomes
    }
    /// Dispatches one op to its applying helper.
    async fn apply_op(&self, ctx: &mut OpContext<'_>, op: &TaskOp) -> TaskOpOutcome {
        match op {
            TaskOp::Create { intent, task_type } => Self::apply_create(ctx, intent, task_type),
            TaskOp::Pause { task_id } => self.apply_pause(ctx, task_id).await,
            TaskOp::Resume { task_id } => self.apply_resume(ctx, task_id).await,
            TaskOp::Cancel { task_id } => self.apply_cancel(ctx, task_id).await,
            TaskOp::Continue {
                depend_task_id,
                event,
                metadata,
                description,
            } => Self::apply_continue(ctx, depend_task_id, event, metadata, description.as_deref()),
            TaskOp::Supplement { task_id, info } => self.apply_supplement(ctx, task_id, info).await,
            TaskOp::Modify { task_id, details } => self.apply_modify(ctx, task_id, details).await,
            TaskOp::Unknown { prompt } => TaskOpOutcome {
                kind: KIND_UNKNOWN,
                task_id: None,
                status: None,
                applied: false,
                message: prompt.clone(),
                clarification: Some(prompt.clone()),
            },
        }
    }
    /// Creates and stores a task for a create op.
    fn apply_create(ctx: &mut OpContext<'_>, intent: &Intent, task_type: &str) -> TaskOpOutcome {
        let planned = ctx.planner.plan_with_task_type(
            ctx.session_id,
            std::slice::from_ref(intent),
            Some(task_type),
        );
        let Some(mut task) = planned.into_iter().next() else {
            return failed(
                KIND_CREATE,
                None,
                format!("no task planned for intent {}", intent.label()),
            );
        };
        let task_id = task.task_id.clone();
        task_type.clone_into(&mut task.task_type);
        task.priority = ctx.config.default_task_priority;
        task.status = TaskStatus::Submitted;
        task.context_id = Some(format!("{}_{task_id}", ctx.session_id));
        task.inputs = vec![intent.event.clone()];
        task.metadata = intent.metadata.clone();
        match ctx.manager.add_task(task) {
            Ok(()) => applied(
                KIND_CREATE,
                &task_id,
                TaskStatus::Submitted,
                format!("task {task_id} created"),
            ),
            Err(error) => failed(KIND_CREATE, Some(task_id), error.to_string()),
        }
    }
    /// Pauses a task through the control seam.
    async fn apply_pause(&self, ctx: &mut OpContext<'_>, task_id: &str) -> TaskOpOutcome {
        if ctx.manager.task(task_id).is_none() {
            return not_found(KIND_PAUSE, task_id);
        }
        let Some(control) = self.control() else {
            return no_control(KIND_PAUSE, task_id);
        };
        if !control.pause(task_id).await {
            return refused(KIND_PAUSE, task_id, "pause");
        }
        applied(
            KIND_PAUSE,
            task_id,
            TaskStatus::Paused,
            format!("task {task_id} paused"),
        )
    }
    /// Cancels a task through the control seam.
    async fn apply_cancel(&self, ctx: &mut OpContext<'_>, task_id: &str) -> TaskOpOutcome {
        if ctx.manager.task(task_id).is_none() {
            return not_found(KIND_CANCEL, task_id);
        }
        let Some(control) = self.control() else {
            return no_control(KIND_CANCEL, task_id);
        };
        if !control.cancel(task_id).await {
            return refused(KIND_CANCEL, task_id, "cancel");
        }
        applied(
            KIND_CANCEL,
            task_id,
            TaskStatus::Canceled,
            format!("task {task_id} canceled"),
        )
    }
    /// Resumes a paused task through the control seam.
    async fn apply_resume(&self, ctx: &mut OpContext<'_>, task_id: &str) -> TaskOpOutcome {
        let Some(task) = ctx.manager.task(task_id) else {
            return not_found(KIND_RESUME, task_id);
        };
        if task.status != TaskStatus::Paused {
            return TaskOpOutcome {
                status: Some(task.status),
                ..failed(
                    KIND_RESUME,
                    Some(task_id.to_owned()),
                    format!("task {task_id} is not paused"),
                )
            };
        }
        let Some(control) = self.control() else {
            return no_control(KIND_RESUME, task_id);
        };
        if !control.resume(task_id).await {
            return refused(KIND_RESUME, task_id, "resume");
        }
        applied(
            KIND_RESUME,
            task_id,
            TaskStatus::Submitted,
            format!("task {task_id} resumed"),
        )
    }
    /// Creates a task that continues from the dependencies.
    fn apply_continue(
        ctx: &mut OpContext<'_>,
        depend_task_id: &[String],
        event: &Event,
        metadata: &BTreeMap<String, serde_json::Value>,
        description: Option<&str>,
    ) -> TaskOpOutcome {
        let mut dependencies = Vec::with_capacity(depend_task_id.len());
        for depend in depend_task_id {
            let Some(task) = ctx.manager.task(depend) else {
                return not_found(KIND_CONTINUE, depend);
            };
            dependencies.push(task);
        }
        let Some(primary) = dependencies.first().cloned() else {
            return failed(
                KIND_CONTINUE,
                None,
                "continue op has no dependency".to_owned(),
            );
        };
        let text = description
            .map(str::to_owned)
            .or_else(|| primary.description.clone())
            .or_else(|| event.text())
            .unwrap_or_default();
        let mut intent = Intent::create_task(event.clone(), text).with_name(KIND_CONTINUE);
        intent.task_type = Some(primary.task_type.clone());
        let planned = ctx.planner.plan_with_task_type(
            ctx.session_id,
            std::slice::from_ref(&intent),
            Some(&primary.task_type),
        );
        let Some(mut task) = planned.into_iter().next() else {
            return failed(
                KIND_CONTINUE,
                None,
                format!("no task planned to continue from {}", primary.task_id),
            );
        };
        let task_id = task.task_id.clone();
        let mut inputs = Vec::new();
        for dependency in &dependencies {
            inputs.extend(dependency.inputs.iter().cloned());
        }
        inputs.push(event.clone());
        task.inputs = inputs;
        task.priority = ctx.config.default_task_priority;
        task.status = TaskStatus::Submitted;
        task.parent_task_id = Some(primary.task_id.clone());
        task.context_id = Some(
            primary
                .context_id
                .clone()
                .unwrap_or_else(|| format!("{}_{task_id}", ctx.session_id)),
        );
        task.metadata = metadata.clone();
        match ctx.manager.add_task(task) {
            Ok(()) => applied(
                KIND_CONTINUE,
                &task_id,
                TaskStatus::Submitted,
                format!("task {task_id} continued from {}", primary.task_id),
            ),
            Err(error) => failed(KIND_CONTINUE, Some(task_id), error.to_string()),
        }
    }
    /// Appends supplied information to a waiting task and re-schedules it.
    async fn apply_supplement(
        &self,
        ctx: &mut OpContext<'_>,
        task_id: &str,
        info: &str,
    ) -> TaskOpOutcome {
        let Some(mut task) = ctx.manager.task(task_id) else {
            return not_found(KIND_SUPPLEMENT, task_id);
        };
        let Some(control) = self.control() else {
            return no_control(KIND_SUPPLEMENT, task_id);
        };
        if !control.pause(task_id).await {
            return refused(KIND_SUPPLEMENT, task_id, "pause");
        }
        let description = task.description.clone().unwrap_or_default();
        task.description = Some(format!("{description}{SUPPLEMENT_MARKER}{info}"));
        if !ctx.manager.update_task(task) {
            return not_found(KIND_SUPPLEMENT, task_id);
        }
        ctx.manager
            .update_task_status(task_id, TaskStatus::Submitted, false, false, None);
        applied(
            KIND_SUPPLEMENT,
            task_id,
            TaskStatus::Submitted,
            format!("task {task_id} supplemented"),
        )
    }
    /// Replaces a task with a modified copy of itself.
    async fn apply_modify(
        &self,
        ctx: &mut OpContext<'_>,
        task_id: &str,
        details: &str,
    ) -> TaskOpOutcome {
        let Some(original) = ctx.manager.task(task_id) else {
            return not_found(KIND_MODIFY, task_id);
        };
        let Some(control) = self.control() else {
            return no_control(KIND_MODIFY, task_id);
        };
        if !control.cancel(task_id).await {
            return refused(KIND_MODIFY, task_id, "cancel");
        }
        let base = original.description.clone().unwrap_or_default();
        let description = format!("{base}{MODIFY_MARKER}{details}");
        let event = original.inputs.last().cloned().unwrap_or_default();
        let mut intent = Intent::create_task(event, description).with_name(KIND_MODIFY);
        intent.task_type = Some(original.task_type.clone());
        let planned = ctx.planner.plan_with_task_type(
            ctx.session_id,
            std::slice::from_ref(&intent),
            Some(&original.task_type),
        );
        let Some(mut task) = planned.into_iter().next() else {
            return failed(
                KIND_MODIFY,
                Some(task_id.to_owned()),
                format!("no task planned to replace {task_id}"),
            );
        };
        let new_id = task.task_id.clone();
        task.priority = ctx.config.default_task_priority;
        task.status = TaskStatus::Submitted;
        task.parent_task_id = Some(task_id.to_owned());
        task.context_id = Some(
            original
                .context_id
                .clone()
                .unwrap_or_else(|| format!("{}_{new_id}", ctx.session_id)),
        );
        task.inputs.clone_from(&original.inputs);
        task.metadata = original.metadata.clone();
        match ctx.manager.add_task(task) {
            Ok(()) => applied(
                KIND_MODIFY,
                &new_id,
                TaskStatus::Submitted,
                format!("task {task_id} modified into {new_id}"),
            ),
            Err(error) => failed(KIND_MODIFY, Some(new_id), error.to_string()),
        }
    }
}
/// The per-call state [`Reasoner::apply_task_ops`] threads through its helpers.
struct OpContext<'a> {
    /// The shared task store.
    manager: &'a TaskManager,
    /// The planner that mints task ids.
    planner: &'a mut Planner,
    /// The session scope used in planned ids and context identities.
    session_id: &'a str,
    /// The configuration the defaults are read from.
    config: &'a ControllerConfig,
}
/// Returns the trimmed target id of `intent`, when it carries one.
fn target_id(intent: &Intent) -> Option<&str> {
    intent
        .target_task_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
}
/// Returns the trimmed text of an optional field, when it is not blank.
fn optional_text(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(str::to_owned)
}

/// Builds the unknown op of an intent.
fn unknown_op(intent: &Intent) -> TaskOp {
    let prompt = optional_text(intent.clarification_prompt.as_deref())
        .unwrap_or_else(|| DEFAULT_CLARIFICATION.to_owned());
    TaskOp::Unknown { prompt }
}
/// Retypes a switch intent as the create intent of the new task.
fn switch_create_intent(intent: &Intent) -> Intent {
    let mut create = intent.clone();
    create.intent_type = IntentType::CreateTask;
    if create
        .target_task_description
        .as_deref()
        .is_none_or(|text| text.trim().is_empty())
    {
        create.target_task_description = intent.event.text().filter(|text| !text.trim().is_empty());
    }
    create
}
/// Resolves the session scope the planned ids are built on.
fn resolve_scope(session_id: &str, session: Option<&Arc<dyn Session>>) -> String {
    let scope = session_id.trim();
    if scope.is_empty() {
        session.map_or_else(String::new, |session| session.id().to_owned())
    } else {
        scope.to_owned()
    }
}
/// Builds an outcome.
fn outcome(
    kind: &'static str,
    task_id: Option<String>,
    status: Option<TaskStatus>,
    applied: bool,
    message: String,
) -> TaskOpOutcome {
    TaskOpOutcome {
        kind,
        task_id,
        status,
        applied,
        message,
        clarification: None,
    }
}
/// Builds an unapplied outcome without a status.
fn failed(kind: &'static str, task_id: Option<String>, message: String) -> TaskOpOutcome {
    outcome(kind, task_id, None, false, message)
}
/// Builds the unapplied outcome of a missing task.
fn not_found(kind: &'static str, task_id: &str) -> TaskOpOutcome {
    failed(
        kind,
        Some(task_id.to_owned()),
        format!("task {task_id} not found"),
    )
}
/// Builds the unapplied outcome of a missing control seam.
fn no_control(kind: &'static str, task_id: &str) -> TaskOpOutcome {
    failed(
        kind,
        Some(task_id.to_owned()),
        "no task control installed".to_owned(),
    )
}
/// Builds the unapplied outcome of a control call that refused the request.
fn refused(kind: &'static str, task_id: &str, action: &str) -> TaskOpOutcome {
    failed(
        kind,
        Some(task_id.to_owned()),
        format!("task {task_id} {action} refused"),
    )
}
/// Builds an applied outcome carrying the resulting status.
fn applied(
    kind: &'static str,
    task_id: &str,
    status: TaskStatus,
    message: String,
) -> TaskOpOutcome {
    outcome(kind, Some(task_id.to_owned()), Some(status), true, message)
}
#[cfg(test)]
mod tests;
