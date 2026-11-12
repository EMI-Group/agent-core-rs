//! Task-loop event-handler hooks.
//!
//! Ports `openjiuwen/harness/task_loop/task_loop_event_handler.py`: the driver
//! calls one hook per observable loop moment, and every hook has a no-op default
//! so an embedder only implements what it needs.
//!
//! Hook order inside one round is fixed:
//! `prepare_round` → `handle_input` → `handle_interaction` (one call per drained
//! steering input) → exactly one of `handle_task_completion` /
//! `handle_task_failed` / `on_abort` → `after_round`.

use async_trait::async_trait;
use parking_lot::Mutex;

use openjiuwen_common::error::Result;

use crate::schema::{ActiveInteractionRound, RoundOutcome};
use crate::task_loop::controller::RoundId;
use crate::task_loop::queues::{LoopQueues, QueuedInput};

/// Observation hooks invoked by the task-loop controller.
///
/// Every hook has a default `Ok(())` implementation. A hook that returns `Err`
/// never panics and never escapes the controller: the round becomes
/// `RoundOutcome::failed("handler_error", message)` (`after_round` failures are
/// only logged).
#[async_trait]
pub trait TaskLoopEventHandler: Send + Sync {
    /// Called before the round starts, after the active round was recorded.
    ///
    /// # Errors
    ///
    /// Any error aborts the round with the `handler_error` code.
    async fn prepare_round(
        &self,
        _round_id: RoundId,
        _round: &ActiveInteractionRound,
    ) -> Result<()> {
        Ok(())
    }

    /// Called after [`Self::prepare_round`], before the model-facing work runs.
    ///
    /// # Errors
    ///
    /// Any error aborts the round with the `handler_error` code.
    async fn handle_input(
        &self,
        _round_id: RoundId,
        _round: &ActiveInteractionRound,
    ) -> Result<()> {
        Ok(())
    }

    /// Called once per steering input drained for this round.
    ///
    /// # Errors
    ///
    /// Any error turns the round into a `handler_error` failure. The remaining
    /// steering inputs of the round are still dispatched exactly once.
    async fn handle_interaction(&self, _round_id: RoundId, _input: QueuedInput) -> Result<()> {
        Ok(())
    }

    /// Called when the round produced no error code.
    ///
    /// # Errors
    ///
    /// Any error turns a successful round into a `handler_error` failure.
    async fn handle_task_completion(
        &self,
        _round_id: RoundId,
        _round: &ActiveInteractionRound,
        _outcome: &RoundOutcome,
    ) -> Result<()> {
        Ok(())
    }

    /// Called when the round produced an `error_code`.
    ///
    /// # Errors
    ///
    /// Hook errors are logged; the round's own error code is preserved.
    async fn handle_task_failed(
        &self,
        _round_id: RoundId,
        _round: &ActiveInteractionRound,
        _outcome: &RoundOutcome,
    ) -> Result<()> {
        Ok(())
    }

    /// Called instead of the completion/failure hooks for an aborted round.
    ///
    /// # Errors
    ///
    /// Hook errors are logged; the `aborted` code is preserved.
    async fn on_abort(&self, _round_id: RoundId, _round: &ActiveInteractionRound) -> Result<()> {
        Ok(())
    }

    /// Called last, for every round, whatever its outcome.
    ///
    /// # Errors
    ///
    /// Hook errors are logged only.
    async fn after_round(
        &self,
        _round_id: RoundId,
        _round: &ActiveInteractionRound,
        _outcome: &RoundOutcome,
    ) -> Result<()> {
        Ok(())
    }
}

/// Handler that forwards steering inputs to a [`LoopQueues`] and records the
/// last terminal outcome.
#[derive(Debug, Default)]
pub struct DefaultTaskLoopEventHandler {
    queues: Option<LoopQueues>,
    last_outcome: Mutex<Option<RoundOutcome>>,
    last_error: Mutex<Option<String>>,
}

impl DefaultTaskLoopEventHandler {
    /// Builds a handler with no steering queue attached.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Attaches the steering queue that [`Self::handle_interaction`] forwards to.
    #[must_use]
    pub fn with_queues(mut self, queues: LoopQueues) -> Self {
        self.queues = Some(queues);
        self
    }

    /// The attached steering queue, when one was configured.
    #[must_use]
    pub fn queues(&self) -> Option<&LoopQueues> {
        self.queues.as_ref()
    }

    /// The outcome recorded by the last terminal hook.
    #[must_use]
    pub fn last_outcome(&self) -> Option<RoundOutcome> {
        self.last_outcome.lock().clone()
    }

    /// The error message recorded by the last failing terminal hook.
    #[must_use]
    pub fn last_error(&self) -> Option<String> {
        self.last_error.lock().clone()
    }

    /// Records a terminal outcome and, when it failed, its message.
    fn record(&self, outcome: &RoundOutcome) {
        if let Some(message) = outcome.error_message.clone() {
            *self.last_error.lock() = Some(message);
        }
        *self.last_outcome.lock() = Some(outcome.clone());
    }
}

#[async_trait]
impl TaskLoopEventHandler for DefaultTaskLoopEventHandler {
    async fn handle_interaction(&self, _round_id: RoundId, input: QueuedInput) -> Result<()> {
        if let Some(queues) = &self.queues {
            queues.push_steer(input);
        }
        Ok(())
    }

    async fn handle_task_completion(
        &self,
        _round_id: RoundId,
        _round: &ActiveInteractionRound,
        outcome: &RoundOutcome,
    ) -> Result<()> {
        self.record(outcome);
        Ok(())
    }

    async fn handle_task_failed(
        &self,
        _round_id: RoundId,
        _round: &ActiveInteractionRound,
        outcome: &RoundOutcome,
    ) -> Result<()> {
        self.record(outcome);
        Ok(())
    }

    async fn on_abort(&self, _round_id: RoundId, _round: &ActiveInteractionRound) -> Result<()> {
        self.record(&RoundOutcome::abort());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::{DefaultTaskLoopEventHandler, TaskLoopEventHandler};
    use crate::schema::{ActiveInteractionRound, RoundOutcome, RoundWorkItem};
    use crate::task_loop::queues::LoopQueues;
    use serde_json::json;
    use std::collections::BTreeMap;

    fn active_round() -> ActiveInteractionRound {
        let mut inputs = BTreeMap::new();
        inputs.insert("query".to_string(), json!("hello"));
        ActiveInteractionRound {
            work: RoundWorkItem::user(None, inputs, false, true),
            task_id: None,
        }
    }

    #[tokio::test]
    async fn default_hooks_are_no_ops() {
        let handler = DefaultTaskLoopEventHandler::new();
        let round = active_round();
        let outcome = RoundOutcome::answer(None);
        handler.prepare_round(1, &round).await.expect("prepare");
        handler.handle_input(1, &round).await.expect("input");
        handler
            .handle_interaction(1, "steer".into())
            .await
            .expect("interaction");
        handler
            .handle_task_completion(1, &round, &outcome)
            .await
            .expect("completion");
        handler
            .after_round(1, &round, &outcome)
            .await
            .expect("after");
        assert!(handler.queues().is_none());
    }

    #[tokio::test]
    async fn interaction_forwards_to_the_steering_queue() {
        let queues = LoopQueues::default();
        let handler = DefaultTaskLoopEventHandler::new().with_queues(queues.clone());
        assert!(handler.queues().is_some());
        handler
            .handle_interaction(1, "steer one".into())
            .await
            .expect("interaction");
        assert_eq!(queues.steering_len(), 1);
        assert_eq!(
            queues.pop_steer().map(|input| input.content().to_string()),
            Some("steer one".to_string())
        );
    }

    #[tokio::test]
    async fn terminal_hooks_record_outcome_and_error() {
        let handler = DefaultTaskLoopEventHandler::new();
        let round = active_round();
        let outcome = RoundOutcome::answer(None);
        handler
            .handle_task_completion(1, &round, &outcome)
            .await
            .expect("completion");
        assert_eq!(handler.last_outcome(), Some(outcome));
        assert!(handler.last_error().is_none());

        let failed = RoundOutcome::failed("round_execution_error", "boom");
        handler
            .handle_task_failed(2, &round, &failed)
            .await
            .expect("failure");
        assert_eq!(handler.last_outcome(), Some(failed));
        assert_eq!(handler.last_error(), Some("boom".to_string()));

        handler.on_abort(3, &round).await.expect("abort");
        let recorded = handler.last_outcome().expect("abort outcome");
        assert_eq!(recorded.error_code.as_deref(), Some("aborted"));
    }
}
