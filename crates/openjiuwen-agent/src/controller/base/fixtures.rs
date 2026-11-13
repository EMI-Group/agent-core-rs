//! Test doubles shared by the controller-loop test modules.
//!
//! The accessor tests in `mod.rs`, the persistence tests in `state`, and the
//! routing tests in `tests` all wire the same controller, so the executor, the
//! event handler, the recognizer, and the turn helpers live here once.
use super::{BaseController, Controller, EventHandler, EventHandlerOutcome};
use crate::card::AgentCard;
use crate::controller::config::ControllerConfig;
use crate::controller::executors::{TaskContext, TaskExecutor};
use crate::controller::intent::IntentRecognizer;
use crate::controller::schema::{ControllerOutput, Event, Intent, Task, TaskResult};
use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_session::{AgentSession, Session};
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
/// Description reported by [`StubExecutor`].
pub(crate) const STUB_DESCRIPTION: &str = "stub executor";
/// Name reported by [`StubHandler`] and [`CountingRecognizer`].
pub(crate) const STUB_NAME: &str = "stub";

/// Executor that records its calls and answers with a fixed text.
pub(crate) struct StubExecutor {
    task_type: &'static str,
    text: &'static str,
    pub(crate) calls: Arc<AtomicUsize>,
}

impl StubExecutor {
    pub(crate) fn new(task_type: &'static str, text: &'static str) -> (Self, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        (
            Self {
                task_type,
                text,
                calls: Arc::clone(&calls),
            },
            calls,
        )
    }
}

#[async_trait]
impl TaskExecutor for StubExecutor {
    fn task_type(&self) -> &str {
        self.task_type
    }

    fn description(&self) -> &str {
        STUB_DESCRIPTION
    }

    async fn execute(&self, _task: &Task, _ctx: &TaskContext<'_>) -> Result<TaskResult> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(TaskResult::text(self.text))
    }
}

/// Handler that answers every event with one fixed outcome.
pub(crate) struct StubHandler {
    pub(crate) outcome: EventHandlerOutcome,
}

#[async_trait]
impl EventHandler for StubHandler {
    fn name(&self) -> &str {
        STUB_NAME
    }

    async fn handle(&self, _event: &Event, _ctx: &TaskContext<'_>) -> Result<EventHandlerOutcome> {
        Ok(self.outcome.clone())
    }
}

/// Recognizer that answers the first call with one intent and every later
/// call with two distinct intents.
pub(crate) struct CountingRecognizer {
    pub(crate) calls: AtomicUsize,
}

#[async_trait]
impl IntentRecognizer for CountingRecognizer {
    fn name(&self) -> &str {
        STUB_NAME
    }

    async fn recognize(&self, event: &Event) -> Result<Vec<Intent>> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        let count = if call == 0 { 1 } else { 2 };
        Ok((0..count)
            .map(|index| {
                Intent::create_task(event.clone(), format!("task {call}-{index}"))
                    .with_name(format!("t{call}{index}"))
            })
            .collect())
    }
}

pub(crate) fn test_session() -> Arc<dyn Session> {
    Arc::new(AgentSession::new("s1"))
}

pub(crate) fn controller_with_executor() -> (BaseController, Arc<AtomicUsize>) {
    let controller = BaseController::new(AgentCard::new("ctrl"), ControllerConfig::new());
    let (executor, calls) = StubExecutor::new("agent", "done");
    controller.add_executor("agent", Arc::new(executor));
    (controller, calls)
}

/// Invokes `controller` with an empty query after queueing one control event
/// that names `task_id`.
pub(crate) async fn control_turn(
    controller: &BaseController,
    session: &Arc<dyn Session>,
    task_id: &str,
    text: &str,
) -> ControllerOutput {
    let mut event = Event::user_input(text);
    event.task_id = Some(task_id.to_owned());
    assert!(controller.push_event(event));
    Controller::invoke(controller, json!({"query": ""}), Some(Arc::clone(session)))
        .await
        .unwrap()
}
