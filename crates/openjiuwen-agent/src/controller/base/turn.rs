//! The single-turn pipeline: process the queued events, route the intents, plan
//! and dispatch the tasks, and aggregate the runs into one output.
//!
//! A turn reads the events queued when it starts (never a background
//! scheduler's), recognizes intents per event, routes the planned task type
//! once per turn, dispatches the batch through the task scheduler, and folds the
//! runs into a controller output. The `stream` seam wraps the same turn in a
//! channel: a `processing` sentinel first, the turn frames next, and one final
//! frame last.

/// Clarification text used when an unknown-task intent carries no prompt.
const CLARIFICATION: &str = "Please clarify your request.";
/// Reply of a paused task control intent.
const PAUSED_REPLY: &str = "paused";
/// Reply of a cancelled task control intent.
const CANCELED_REPLY: &str = "canceled";

use super::{BaseController, Controller, EventHandlerOutcome, TurnState};
use crate::card::AgentCard;
use crate::controller::config::ControllerConfig;
use crate::controller::executors::{TaskContext, TaskRun};
use crate::controller::schema::{
    frames_text, ControllerOutput, ControllerOutputChunk, ControllerOutputPayload, DataFrame,
    Event, EventType, Intent, IntentType, OutputKind, Task, TaskStatus,
};
use crate::run::resolve_session;
use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_common::error::Result;
use openjiuwen_session::Session;
use serde_json::{json, Value};
use std::sync::Arc;
use tracing::{debug, warn};

impl TurnState {
    /// Builds the routing state of a turn over `events`.
    fn new(events: &[Event]) -> Self {
        Self {
            query: first_query(events),
            resolved: false,
            task_type: None,
        }
    }
}

impl BaseController {
    /// Runs one turn: queue the event, recognize, plan, dispatch, aggregate.
    async fn invoke_turn(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> Result<ControllerOutput> {
        let event = Event::from_user_inputs(&inputs)?;
        let run = resolve_session(
            session,
            self.card().id(),
            Some(self.card().name()),
            None,
            None,
        )
        .await?;
        let session = Arc::clone(run.session());
        self.start();
        self.bind_session(Arc::clone(&session)).await;
        if self.config().enable_task_persistence {
            self.load_state(session.as_ref()).await?;
        }
        if !self.push_event(event) {
            warn!(
                controller_id = self.card().id(),
                "controller event queue is full; dropping the inbound event"
            );
        }
        let events = self.drain_events();
        let mut turn = TurnState::new(&events);
        let mut output: Option<ControllerOutput> = None;
        for event in &events {
            if let Some(produced) = self.process_event(event, &session, &mut turn).await? {
                output = Some(produced);
            }
        }
        let output = output.unwrap_or_else(|| self.fallback_output());
        self.save_state(session.as_ref()).await?;
        // An idle session gets Python's `all_tasks_processed` sentinel; the
        // loop is only present when this controller started one, and the
        // session must still be bound for the loop's session lookup.
        if let Some(scheduler) = self.scheduler_loop() {
            scheduler.ensure_completion_signal(session.id()).await;
        }
        self.unbind_session(session.id()).await;
        run.finish().await;
        Ok(output)
    }

    /// Streams one turn: a `processing` frame, the turn frames, and one final
    /// frame.
    fn stream_turn(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> BoxStream<'static, Result<ControllerOutputChunk>> {
        let controller = self.clone();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Result<ControllerOutputChunk>>();
        tokio::spawn(async move {
            let _ = tx.send(Ok(ControllerOutputChunk::new(
                0,
                ControllerOutputPayload::processing(),
            )));
            match controller.invoke_turn(inputs, session).await {
                Ok(output) => {
                    let mut index = 1_i64;
                    for frame in &output.frames {
                        let mut frame = frame.clone();
                        frame.index = index;
                        frame.last_chunk = false;
                        index += 1;
                        if tx.send(Ok(frame)).is_err() {
                            return;
                        }
                    }
                    let kind = match output.status {
                        TaskStatus::InputRequired => OutputKind::TaskInteraction,
                        TaskStatus::Failed => OutputKind::TaskFailed,
                        _ => OutputKind::TaskCompletion,
                    };
                    let last = ControllerOutputChunk::last(
                        index,
                        ControllerOutputPayload::text(kind, output.output),
                    );
                    let _ = tx.send(Ok(last));
                }
                Err(error) => {
                    let _ = tx.send(Err(error));
                }
            }
        });
        futures::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        })
        .boxed()
    }
}

impl BaseController {
    /// Produces the turn output of one drained event.
    ///
    /// `Ok(None)` means the event contributed nothing (an ignored handler
    /// decision or an input that planned no task), so an earlier event's output
    /// stays the turn output.
    async fn process_event(
        &self,
        event: &Event,
        session: &Arc<dyn Session>,
        turn: &mut TurnState,
    ) -> Result<Option<ControllerOutput>> {
        let handler = self.inner.handler.read().clone();
        if let Some(handler) = handler {
            let ctx = TaskContext::new(Some(Arc::clone(session)), self.card().id());
            return match handler.handle(event, &ctx).await? {
                EventHandlerOutcome::Output(output) => Ok(Some(*output)),
                EventHandlerOutcome::Ignored => Ok(None),
                EventHandlerOutcome::Intents(intents) => {
                    self.handle_intents(event, intents, session, turn).await
                }
            };
        }
        if matches!(event.event_type, EventType::Input | EventType::FollowUp) {
            let intents = if self.config().enable_intent_recognition {
                let recognizer = self.inner.recognizer.read().clone();
                recognizer.recognize(event).await?
            } else {
                vec![Intent::default_for(event.clone())]
            };
            return self.handle_intents(event, intents, session, turn).await;
        }
        Ok(self.signal_output(event))
    }

    /// Routes recognized intents: clarification, task control, or new tasks.
    async fn handle_intents(
        &self,
        event: &Event,
        intents: Vec<Intent>,
        session: &Arc<dyn Session>,
        turn: &mut TurnState,
    ) -> Result<Option<ControllerOutput>> {
        let config = self.config();
        let mut accepted: Vec<Intent> = intents
            .into_iter()
            .filter(|intent| config.accepts(intent.intent_type))
            .collect();
        if config.max_intents > 0 {
            accepted.truncate(config.max_intents);
        }
        if accepted.is_empty() && matches!(event.event_type, EventType::Input | EventType::FollowUp)
        {
            accepted.push(Intent::default_for(event.clone()));
        }
        if accepted.is_empty() {
            return Ok(None);
        }
        let mut planned: Vec<Intent> = Vec::new();
        let mut dispatch_queue: Vec<Task> = Vec::new();
        let mut control: Option<ControllerOutput> = None;
        for intent in accepted {
            match intent.intent_type {
                IntentType::UnknownTask => return Ok(Some(clarification_output(&intent))),
                IntentType::PauseTask | IntentType::CancelTask => {
                    let status = if intent.intent_type == IntentType::PauseTask {
                        TaskStatus::Paused
                    } else {
                        TaskStatus::Canceled
                    };
                    control = Some(match self.set_task_status(&intent, status) {
                        Some(output) => output,
                        None => not_found_output(&intent),
                    });
                }
                IntentType::ResumeTask => match self.resume_task(&intent) {
                    Some(task) => dispatch_queue.push(task),
                    None => control = Some(not_found_output(&intent)),
                },
                _ => planned.push(intent),
            }
        }
        if !planned.is_empty() {
            let task_type = self.route(turn, &planned).await?;
            dispatch_queue.extend(self.plan_tasks(&planned, task_type.as_deref(), session.id()));
        }
        if dispatch_queue.is_empty() {
            return Ok(control);
        }
        Ok(Some(self.dispatch_runs(dispatch_queue, session).await))
    }

    /// Asks the router once per turn for the planned task type.
    async fn route(&self, turn: &mut TurnState, intents: &[Intent]) -> Result<Option<String>> {
        if turn.resolved {
            return Ok(turn.task_type.clone());
        }
        let router = self.inner.router.read().clone();
        let has_choice = self.inner.registry.read().len() >= 2;
        let route = match router {
            Some(router) if has_choice => router
                .select(&turn.query, intents)
                .await?
                .filter(|task_type| !task_type.is_empty()),
            _ => None,
        };
        turn.resolved = true;
        turn.task_type.clone_from(&route);
        Ok(route)
    }

    /// Plans one task per valid, distinct intent.
    fn plan_tasks(
        &self,
        intents: &[Intent],
        task_type: Option<&str>,
        session_id: &str,
    ) -> Vec<Task> {
        self.inner
            .planner
            .write()
            .plan_with_task_type(session_id, intents, task_type)
    }

    /// Moves the persisted task targeted by a control intent to `status`.
    fn set_task_status(&self, intent: &Intent, status: TaskStatus) -> Option<ControllerOutput> {
        let id = intent.target_task_id.as_deref()?;
        let manager = self.task_manager();
        manager.task(id)?;
        manager.update_task_status(id, status, false, false, None);
        let reply = match status {
            TaskStatus::Canceled => CANCELED_REPLY,
            _ => PAUSED_REPLY,
        };
        Some(
            ControllerOutput::new(format!("task {id} {reply}"), status)
                .with_task_id(id)
                .with_data(json!({"task_id": id, "status": status.as_str()})),
        )
    }

    /// Returns the persisted task targeted by a resume intent, re-queued.
    ///
    /// The task is claimed for this turn's synchronous dispatch, so the
    /// background scheduler cannot pick up the same run in parallel; the run's
    /// outcome is upserted back into the store by
    /// [`dispatch_runs`](Self::dispatch_runs).
    fn resume_task(&self, intent: &Intent) -> Option<Task> {
        let id = intent.target_task_id.as_deref()?;
        let manager = self.task_manager();
        let mut task = manager.task(id)?;
        manager.update_task_status(id, TaskStatus::Submitted, false, false, None);
        let _ = manager.claim_task(id);
        task.status = TaskStatus::Submitted;
        Some(task)
    }

    /// Dispatches `tasks` and aggregates the runs into one output.
    async fn dispatch_runs(
        &self,
        tasks: Vec<Task>,
        session: &Arc<dyn Session>,
    ) -> ControllerOutput {
        let ctx = TaskContext::new(Some(Arc::clone(session)), self.card().id());
        let runs = self.scheduler().dispatch(tasks, &ctx).await;
        let config = self.config();
        let mut frames = Vec::new();
        let mut index = 0_i64;
        let mut texts = Vec::new();
        let mut entries = Vec::new();
        for run in &runs {
            frames.extend(run.frames(index));
            index += 1;
            if run.status() == TaskStatus::Completed {
                let text = run.text();
                if !text.is_empty() {
                    texts.push(text);
                }
            }
            entries.push(json!({
                "task_id": run.task.task_id,
                "status": run.status().as_str(),
                "output": run.text(),
            }));
        }
        {
            // The turn's batch enters the store only after it ran, so the
            // background scheduler never sees a submitted task of this turn.
            let manager = self.task_manager();
            for run in &runs {
                let task = run.task.clone();
                if manager.update_task(task.clone()) {
                    continue;
                }
                if let Err(error) = manager.add_task(task) {
                    warn!(
                        task_id = %run.task.task_id,
                        %error,
                        "cannot store the dispatched task"
                    );
                }
            }
        }
        let status = aggregate_status(&runs);
        let mut text = texts.join("\n\n");
        if text.is_empty() {
            text = config.default_response.text.clone().unwrap_or_default();
        }
        if !config.suppress_completion_signal {
            let kind = match status {
                TaskStatus::InputRequired => OutputKind::TaskInteraction,
                TaskStatus::Failed => OutputKind::TaskFailed,
                _ => OutputKind::TaskCompletion,
            };
            frames.push(ControllerOutputChunk::last(
                index,
                ControllerOutputPayload::text(kind, text.clone()),
            ));
        }
        let mut output = ControllerOutput::new(text, status)
            .with_data(json!({"tasks": entries}))
            .with_frames(frames);
        if let Some(run) = runs.last() {
            output = output.with_task_id(run.task.task_id.clone());
        }
        output
    }

    /// Builds the output of a task signal event and records it on the task.
    fn signal_output(&self, event: &Event) -> Option<ControllerOutput> {
        let task_id = event.task_id.clone().unwrap_or_default();
        let (status, text, kind, data) = match event.event_type {
            EventType::TaskCompletion => {
                let text = event
                    .text()
                    .filter(|text| !text.is_empty())
                    .unwrap_or_else(|| format!("task {task_id} completed"));
                (
                    TaskStatus::Completed,
                    text,
                    OutputKind::TaskCompletion,
                    event.task_result.clone(),
                )
            }
            EventType::TaskFailed => {
                let text = event
                    .error_message
                    .clone()
                    .filter(|text| !text.is_empty())
                    .or_else(|| event.text().filter(|text| !text.is_empty()))
                    .unwrap_or_else(|| format!("task {task_id} failed"));
                (TaskStatus::Failed, text, OutputKind::TaskFailed, Vec::new())
            }
            EventType::TaskInteraction => {
                let text = frames_text(&event.interaction)
                    .filter(|text| !text.is_empty())
                    .unwrap_or_else(|| format!("task {task_id} requires input"));
                (
                    TaskStatus::InputRequired,
                    text,
                    OutputKind::TaskInteraction,
                    event.interaction.clone(),
                )
            }
            EventType::Input | EventType::FollowUp => return None,
        };
        let frames = if data.is_empty() {
            vec![DataFrame::text(text.clone())]
        } else {
            data
        };
        let chunk = ControllerOutputChunk::last(0, ControllerOutputPayload::new(kind, frames));
        if !task_id.is_empty() {
            self.record_signal(&task_id, status, event, &text, &chunk);
        }
        let mut output = ControllerOutput::new(text, status).with_frames(vec![chunk]);
        if !task_id.is_empty() {
            output = output.with_task_id(task_id);
        }
        Some(output)
    }

    /// Folds a task signal event into the persisted task it names.
    fn record_signal(
        &self,
        task_id: &str,
        status: TaskStatus,
        event: &Event,
        text: &str,
        chunk: &ControllerOutputChunk,
    ) {
        let manager = self.task_manager();
        let Some(mut task) = manager.task(task_id) else {
            debug!(
                task_id,
                "controller signal names a task that is not persisted"
            );
            return;
        };
        task.status = status;
        if status == TaskStatus::Failed {
            task.error_message = event
                .error_message
                .clone()
                .or_else(|| Some(text.to_owned()));
        }
        if status == TaskStatus::InputRequired && task.input_required_fields.is_none() {
            task.input_required_fields = Some(json!({"fields": []}));
        }
        task.outputs.push(chunk.clone());
        let _ = manager.update_task(task);
    }

    /// Output of a turn that produced nothing: the configured default response.
    fn fallback_output(&self) -> ControllerOutput {
        let text = self.config().default_response.text.unwrap_or_default();
        ControllerOutput::new(text, TaskStatus::Completed)
    }
}

#[async_trait]
impl Controller for BaseController {
    fn card(&self) -> &AgentCard {
        BaseController::card(self)
    }

    fn config(&self) -> ControllerConfig {
        BaseController::config(self)
    }

    async fn invoke(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> Result<ControllerOutput> {
        self.invoke_turn(inputs, session).await
    }

    fn stream(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> BoxStream<'static, Result<ControllerOutputChunk>> {
        self.stream_turn(inputs, session)
    }
}

/// First non-empty event text of a turn, or `""`.
fn first_query(events: &[Event]) -> String {
    events
        .iter()
        .find_map(|event| event.text().filter(|text| !text.is_empty()))
        .unwrap_or_default()
}

/// Status of a dispatched batch: interaction first, then completion, then
/// failure; an empty batch counts as completed.
fn aggregate_status(runs: &[TaskRun]) -> TaskStatus {
    if runs
        .iter()
        .any(|run| run.status() == TaskStatus::InputRequired)
    {
        TaskStatus::InputRequired
    } else if runs.iter().any(|run| run.status() == TaskStatus::Completed) {
        TaskStatus::Completed
    } else if runs.iter().any(|run| run.status() == TaskStatus::Failed) {
        TaskStatus::Failed
    } else {
        TaskStatus::Completed
    }
}

/// Builds the clarification output of an unknown-task intent.
fn clarification_output(intent: &Intent) -> ControllerOutput {
    let prompt = intent
        .clarification_prompt
        .clone()
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| CLARIFICATION.to_owned());
    ControllerOutput::new(prompt.clone(), TaskStatus::InputRequired)
        .with_data(json!({"clarification_prompt": prompt}))
}

/// Builds the failed output of a control intent whose target is unknown.
fn not_found_output(intent: &Intent) -> ControllerOutput {
    let id = intent.target_task_id.clone().unwrap_or_default();
    warn!(
        intent = intent.label(),
        task_id = %id,
        "controller intent targets an unknown task"
    );
    ControllerOutput::new(format!("task {id} not found"), TaskStatus::Failed)
        .with_data(json!({"task_id": id}))
}
