//! The background turn machinery of one sub-agent instance.
//!
//! One [`TurnRunner`] task drains the queued queries of a single instance serially: it takes a
//! concurrency permit, marks the instance `Running`, invokes the child agent under the turn
//! timeout, stores the result, and either starts the next queued query or parks the instance in
//! the `Idle` activity phase. Callers never hold the state lock across an `.await`.
use std::sync::Arc;
use std::time::Duration;

use openjiuwen_agent::{Agent, AgentOutput, ResultType};
use serde_json::json;
use tokio::sync::Semaphore;

use super::clock::Clock;
use super::lifecycle::{LifecycleStage, SubagentLifecycleEvent};
use super::quota::SubagentActivity;
use super::transcript::TranscriptEntry;
use super::{lifecycle_api, Shared, SubagentStatus};

/// One background turn worker; it runs queued queries of one instance serially.
pub(super) struct TurnRunner {
    /// Runtime state shared with the controller.
    pub(super) shared: Arc<Shared>,
    /// Bounds concurrent turns across instances.
    pub(super) semaphore: Arc<Semaphore>,
    /// Source of the activity timestamps.
    pub(super) clock: Arc<dyn Clock>,
    /// Per-turn timeout.
    pub(super) turn_timeout: Duration,
    /// Instance this runner owns.
    pub(super) subagent_id: String,
    /// Turn generation the runner was started with.
    pub(super) generation: u64,
    /// Child agent to invoke.
    pub(super) agent: Arc<dyn Agent>,
    /// Child session id passed to every invocation.
    pub(super) sub_session_id: String,
}

impl TurnRunner {
    /// Drains the queued queries, starting with `first_query`.
    pub(super) async fn run(self, first_query: String) {
        let mut next_query = Some(first_query);
        while let Some(query) = next_query.take() {
            let permit = match self.semaphore.acquire().await {
                Ok(permit) => permit,
                Err(error) => {
                    tracing::debug!(%error, "subagent turn semaphore closed");
                    return;
                }
            };
            if !begin_turn(
                &self.shared,
                self.clock.as_ref(),
                &self.subagent_id,
                self.generation,
            ) {
                return;
            }
            let inputs = json!({"query": query, "conversation_id": self.sub_session_id});
            let result =
                tokio::time::timeout(self.turn_timeout, self.agent.invoke(inputs, None)).await;
            drop(permit);
            let (status, output, error) = map_turn_result(result, self.turn_timeout);
            let (next, ready) = finish_turn(
                &self.shared,
                self.clock.as_ref(),
                &self.subagent_id,
                self.generation,
                status,
                output,
                error,
            );
            // `Ready` fires only once the instance is parked idle, and only after the state
            // update above; the hook runs without the state lock held.
            if let Some(event) = ready {
                lifecycle_api::run_stage(&self.shared, LifecycleStage::Ready, [event]).await;
            }
            next_query = next;
        }
    }
}

/// Marks the instance `Running` if the turn still owns it.
pub(super) fn begin_turn(shared: &Shared, clock: &dyn Clock, id: &str, generation: u64) -> bool {
    let now_ms = clock.now_ms();
    let started = {
        let mut state = shared.state.lock();
        match state.instances.get_mut(id) {
            Some(instance)
                if instance.generation == generation
                    && instance.status != SubagentStatus::Closed =>
            {
                instance.status = SubagentStatus::Running;
                instance.touch(now_ms, SubagentActivity::Running);
                instance.record(TranscriptEntry::status(
                    id,
                    SubagentStatus::Running.as_str(),
                    now_ms,
                ));
                let task_id = format!("{id}#{generation}");
                instance.activity_stream.begin_turn(&task_id, now_ms);
                instance.activity_stream.running(now_ms);
                true
            }
            _ => false,
        }
    };
    if started {
        shared.notify.notify_waiters();
        tracing::debug!(target: "openjiuwen_harness::subagents", subagent_id = %id, status = "running", "subagent turn started");
    }
    started
}

/// Stores the turn result and returns the next queued query, if any, plus the `Ready` lifecycle
/// snapshot when the worker parks the instance idle.
///
/// A turn that leaves queued work keeps the instance `Running` (the same worker keeps going and
/// the next [`begin_turn`] publishes the new turn's `Running` frame); the last turn parks it in
/// the `Idle` activity phase and publishes the terminal outcome frame whatever its final status.
pub(super) fn finish_turn(
    shared: &Shared,
    clock: &dyn Clock,
    id: &str,
    generation: u64,
    status: SubagentStatus,
    output: Option<String>,
    error: Option<String>,
) -> (Option<String>, Option<SubagentLifecycleEvent>) {
    let now_ms = clock.now_ms();
    let next = {
        let mut state = shared.state.lock();
        let instance = match state.instances.get_mut(id) {
            Some(instance)
                if instance.generation == generation
                    && instance.status != SubagentStatus::Closed =>
            {
                instance
            }
            _ => return (None, None),
        };
        instance.output = output;
        instance.error = error;
        if let Some(query) = instance.pending.pop_front() {
            instance.status = SubagentStatus::PendingInit;
            instance.touch(now_ms, SubagentActivity::Running);
            instance.record(TranscriptEntry::input(id, query.as_str(), now_ms));
            (Some(query), None)
        } else {
            instance.status = status;
            instance.running = false;
            instance.touch(now_ms, SubagentActivity::Idle);
            let entry = match status {
                SubagentStatus::Completed => TranscriptEntry::completion(
                    id,
                    instance.output.clone().unwrap_or_default(),
                    now_ms,
                ),
                SubagentStatus::Errored => {
                    TranscriptEntry::error(id, instance.error.clone().unwrap_or_default(), now_ms)
                }
                other => TranscriptEntry::status(id, other.as_str(), now_ms),
            };
            instance.record(entry);
            instance.activity_stream.idle(now_ms);
            match status {
                SubagentStatus::Completed => {
                    let output = instance.output.clone().unwrap_or_default();
                    instance.activity_stream.completed(&output, now_ms);
                }
                SubagentStatus::Errored => {
                    let message = instance.error.clone().unwrap_or_default();
                    instance.activity_stream.failed(&message, now_ms);
                }
                SubagentStatus::Interrupted => {
                    instance.activity_stream.failed("interrupted", now_ms);
                }
                _ => {}
            }
            tracing::debug!(target: "openjiuwen_harness::subagents", subagent_id = %id, status = status.as_str(), "subagent turn finished");
            let ready = instance.lifecycle_event(LifecycleStage::Ready, None, now_ms);
            (None, Some(ready))
        }
    };
    shared.notify.notify_waiters();
    next
}

/// Maps one invocation result to the next status and its payload.
pub(super) fn map_turn_result(
    result: std::result::Result<
        openjiuwen_common::error::Result<AgentOutput>,
        tokio::time::error::Elapsed,
    >,
    turn_timeout: Duration,
) -> (SubagentStatus, Option<String>, Option<String>) {
    match result {
        Ok(Ok(output)) => match output.result_type {
            ResultType::Answer => (SubagentStatus::Completed, Some(output.output), None),
            ResultType::Error => (SubagentStatus::Errored, None, Some(output.output)),
            ResultType::Interrupt => (SubagentStatus::Interrupted, None, None),
        },
        Ok(Err(error)) => (SubagentStatus::Errored, None, Some(error.to_string())),
        Err(_elapsed) => (
            SubagentStatus::Errored,
            None,
            Some(format!(
                "subagent turn timed out after {}s",
                turn_timeout.as_secs()
            )),
        ),
    }
}
