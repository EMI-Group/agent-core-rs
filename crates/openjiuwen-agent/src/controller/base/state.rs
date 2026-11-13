//! Task persistence: the controller's task map and the planner id sequence.
//!
//! The state lives in the session under
//! [`TASKS_STATE_KEY`](super::TASKS_STATE_KEY): a turn restores it before
//! dispatching and saves it afterwards, so tasks and the planner's id sequence
//! survive between turns and between controller instances.
use super::BaseController;
use super::TASKS_STATE_KEY;
use crate::controller::schema::Task;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_session::Session;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tracing::warn;

/// The persisted controller state.
#[derive(Debug, Default, Serialize, Deserialize)]
struct TasksState {
    /// Tasks in id order.
    #[serde(default)]
    tasks: Vec<Task>,
    /// Next planner sequence number.
    #[serde(default)]
    next_seq: u64,
}

impl BaseController {
    /// Persists the task map and the planner sequence on `session`.
    ///
    /// Does nothing when
    /// [`enable_task_persistence`](super::ControllerConfig::enable_task_persistence)
    /// is off.
    ///
    /// # Errors
    ///
    /// Propagates the session's state-write error and returns
    /// [`Error::validation`] when the task map cannot be serialized.
    pub async fn save_state(&self, session: &dyn Session) -> Result<()> {
        if !self.config().enable_task_persistence {
            return Ok(());
        }
        let state = TasksState {
            tasks: self.inner.tasks.read().values().cloned().collect(),
            next_seq: self.inner.planner.read().next_seq(),
        };
        let value = serde_json::to_value(&state)
            .map_err(|error| Error::validation(format!("cannot serialize task state: {error}")))?;
        session
            .update_state(json!({ TASKS_STATE_KEY: value }))
            .await
    }

    /// Restores the task map and the planner sequence from `session`.
    ///
    /// The read is tolerant: a missing or malformed state clears the in-memory
    /// task map and restarts the id sequence at one instead of failing.
    ///
    /// # Errors
    ///
    /// Propagates the session's state-read error.
    pub async fn load_state(&self, session: &dyn Session) -> Result<()> {
        let state = session.get_state(TASKS_STATE_KEY).await?;
        let parsed = match state {
            None | Some(Value::Null) => None,
            Some(value) => match serde_json::from_value::<TasksState>(value) {
                Ok(state) => Some(state),
                Err(error) => {
                    warn!(%error, "ignoring an unreadable controller task state");
                    None
                }
            },
        };
        let (tasks, next_seq) = parsed.map_or_else(
            || (Vec::new(), 1),
            |state| (state.tasks, state.next_seq.max(1)),
        );
        let map = tasks
            .into_iter()
            .map(|task| (task.task_id.clone(), task))
            .collect();
        *self.inner.tasks.write() = map;
        self.inner.planner.write().set_next_seq(next_seq);
        Ok(())
    }

    /// Returns the persisted tasks, sorted by task id.
    // `async` because the surface is shared with `save_state`/`load_state`.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    #[must_use]
    pub async fn tasks(&self) -> Vec<Task> {
        self.inner.tasks.read().values().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::AgentCard;
    use crate::controller::base::fixtures::{
        control_turn, test_session, CountingRecognizer, StubExecutor,
    };
    use crate::controller::base::Controller;
    use crate::controller::config::ControllerConfig;
    use crate::controller::intent::{IntentRule, RuleIntentRecognizer};
    use crate::controller::schema::{IntentType, TaskStatus};
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    #[tokio::test]
    async fn state_persists_across_turns_and_instances() {
        let controller = BaseController::with_recognizer(
            AgentCard::new("ctrl"),
            ControllerConfig::new(),
            Arc::new(CountingRecognizer {
                calls: AtomicUsize::new(0),
            }),
        );
        controller.add_executor("agent", Arc::new(StubExecutor::new("agent", "done").0));
        let session = test_session();
        let first = Controller::invoke(
            &controller,
            json!({"query": "one"}),
            Some(Arc::clone(&session)),
        )
        .await
        .unwrap();
        assert_eq!(controller.tasks().await.len(), 1);
        let second = Controller::invoke(
            &controller,
            json!({"query": "two"}),
            Some(Arc::clone(&session)),
        )
        .await
        .unwrap();
        assert_ne!(first.task_id, second.task_id);
        let tasks = controller.tasks().await;
        assert_eq!(tasks.len(), 3);
        let ids: std::collections::BTreeSet<_> =
            tasks.iter().map(|task| task.task_id.clone()).collect();
        assert_eq!(ids.len(), 3);
        let restored = BaseController::new(AgentCard::new("ctrl"), ControllerConfig::new());
        restored.load_state(session.as_ref()).await.unwrap();
        assert_eq!(restored.tasks().await.len(), 3);
    }

    #[tokio::test]
    async fn pause_and_cancel_flip_the_persisted_task_status() {
        let recognizer = Arc::new(RuleIntentRecognizer::with_rules(vec![
            IntentRule::new("pause", IntentType::PauseTask, ["pause"]),
            IntentRule::new("cancel", IntentType::CancelTask, ["cancel"]),
        ]));
        let controller = BaseController::with_recognizer(
            AgentCard::new("ctrl"),
            ControllerConfig::new(),
            recognizer,
        );
        controller.add_executor("agent", Arc::new(StubExecutor::new("agent", "done").0));
        let session = test_session();
        Controller::invoke(
            &controller,
            json!({"query": "build it"}),
            Some(Arc::clone(&session)),
        )
        .await
        .unwrap();
        let task_id = controller.tasks().await[0].task_id.clone();
        let paused = control_turn(&controller, &session, &task_id, "pause it").await;
        assert_eq!(paused.status, TaskStatus::Paused);
        assert_eq!(controller.tasks().await[0].status, TaskStatus::Paused);
        let canceled = control_turn(&controller, &session, &task_id, "cancel it").await;
        assert_eq!(canceled.status, TaskStatus::Canceled);
        assert_eq!(controller.tasks().await[0].status, TaskStatus::Canceled);
        let missing = control_turn(&controller, &session, "nope", "pause it").await;
        assert_eq!(missing.status, TaskStatus::Failed);
        assert!(missing.output.contains("not found"));
    }
}
