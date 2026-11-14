//! Task persistence: the shared task store and the planner id sequence.
//!
//! The state lives in the session under
//! [`TASKS_STATE_KEY`](super::TASKS_STATE_KEY) as the shared manager's
//! [`TaskManagerState`] (`{"tasks": [...], "next_seq": n}`): a turn restores it
//! before dispatching and saves it afterwards, so tasks and the planner's id
//! sequence survive between turns and between controller instances. The
//! snapshot is scoped to the session, so a manager shared by several sessions
//! never leaks one session's tasks into another.
use super::BaseController;
use super::TASKS_STATE_KEY;
use crate::controller::schema::Task;
use crate::controller::task_manager::TaskManagerState;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_session::Session;
use serde_json::{json, Value};
use tracing::warn;

impl BaseController {
    /// Persists this session's tasks and the planner sequence on `session`.
    ///
    /// The wire shape is the manager's own snapshot, with `next_seq` carrying
    /// the planner's id sequence (the manager's ordering sequence is internal
    /// and not restored from the wire). Does nothing when
    /// [`enable_task_persistence`](super::ControllerConfig::enable_task_persistence)
    /// is off.
    ///
    /// # Errors
    ///
    /// Propagates the session's state-write error and returns
    /// [`Error::validation`] when the task snapshot cannot be serialized.
    pub async fn save_state(&self, session: &dyn Session) -> Result<()> {
        if !self.config().enable_task_persistence {
            return Ok(());
        }
        let mut state = self.inner.tasks.snapshot_session(session.id());
        state.next_seq = self.inner.planner.read().next_seq();
        let value = serde_json::to_value(&state)
            .map_err(|error| Error::validation(format!("cannot serialize task state: {error}")))?;
        session
            .update_state(json!({ TASKS_STATE_KEY: value }))
            .await
    }

    /// Restores this session's tasks and the planner sequence from `session`.
    ///
    /// The read is tolerant: a missing, malformed, or unrestorable state clears
    /// the session's tasks in the manager and restarts the id sequence at one
    /// instead of failing. Tasks of other sessions sharing the manager are left
    /// untouched.
    ///
    /// # Errors
    ///
    /// Propagates the session's state-read error.
    pub async fn load_state(&self, session: &dyn Session) -> Result<()> {
        let state = session.get_state(TASKS_STATE_KEY).await?;
        let parsed = match state {
            None | Some(Value::Null) => None,
            Some(value) => match serde_json::from_value::<TaskManagerState>(value) {
                Ok(state) => Some(state),
                Err(error) => {
                    warn!(%error, "ignoring an unreadable controller task state");
                    None
                }
            },
        };
        let next_seq = parsed.as_ref().map_or(1, |state| state.next_seq.max(1));
        if let Err(error) = self
            .inner
            .tasks
            .restore_session(session.id(), parsed.as_ref())
        {
            warn!(
                %error,
                session_id = session.id(),
                "ignoring a controller task state that cannot be restored"
            );
            if let Err(error) = self.inner.tasks.restore_session(session.id(), None) {
                warn!(%error, "cannot clear the controller task state");
            }
        }
        self.inner.planner.write().set_next_seq(next_seq);
        Ok(())
    }

    /// Returns the persisted tasks, in creation order.
    // `async` because the surface is shared with `save_state`/`load_state`.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    #[must_use]
    pub async fn tasks(&self) -> Vec<Task> {
        self.inner.tasks.tasks()
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

    #[tokio::test]
    async fn missing_or_corrupt_state_restarts_the_sequences() {
        let controller = BaseController::new(AgentCard::new("ctrl"), ControllerConfig::new());
        let session = test_session();
        controller.load_state(session.as_ref()).await.unwrap();
        assert_eq!(controller.tasks().await.len(), 0);
        assert_eq!(controller.inner.planner.read().next_seq(), 1);

        session
            .update_state(json!({TASKS_STATE_KEY: {"tasks": "not a list"}}))
            .await
            .unwrap();
        controller.load_state(session.as_ref()).await.unwrap();
        assert_eq!(controller.tasks().await.len(), 0);
        assert_eq!(controller.inner.planner.read().next_seq(), 1);
    }

    #[tokio::test]
    async fn persistence_can_be_disabled() {
        let config = ControllerConfig {
            enable_task_persistence: false,
            ..ControllerConfig::new()
        };
        let controller = BaseController::new(AgentCard::new("ctrl"), config);
        controller
            .task_manager()
            .add_task(Task::new("s1", "t1", "agent"))
            .unwrap();
        let session = test_session();
        controller.save_state(session.as_ref()).await.unwrap();
        assert!(session.get_state(TASKS_STATE_KEY).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn save_state_writes_the_manager_snapshot_and_restores_it() {
        let controller = BaseController::new(AgentCard::new("ctrl"), ControllerConfig::new());
        controller
            .task_manager()
            .add_task(
                Task::new("s1", "t1", "agent")
                    .with_status(TaskStatus::Completed)
                    .with_priority(3),
            )
            .unwrap();
        let session = test_session();
        controller.save_state(session.as_ref()).await.unwrap();
        let state = session
            .get_state(TASKS_STATE_KEY)
            .await
            .unwrap()
            .expect("the save wrote the state");
        assert_eq!(state["tasks"][0]["task_id"], json!("t1"));
        assert_eq!(state["tasks"][0]["status"], json!("completed"));
        assert_eq!(state["next_seq"], json!(1));
        assert_eq!(state.as_object().unwrap().len(), 2);

        let restored = BaseController::new(AgentCard::new("ctrl"), ControllerConfig::new());
        restored.load_state(session.as_ref()).await.unwrap();
        let tasks = restored.tasks().await;
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].task_id, "t1");
        assert_eq!(tasks[0].priority, 3);
    }
}
