//! The turn task that drives one native-harness `DeepAgent::stream`.
//!
//! [`spawn_turn`] starts one tokio task per harness turn. The task consumes the
//! kernel frames, maps them to protocol events (see [`super::events`]), honours
//! pause/abort/stop at frame boundaries, chains queued follow-up inputs, and
//! always publishes one terminal `TurnLifecycle` event carrying the
//! [`TurnResult`].
//!
//! Pause and abort are cooperative here because the kernel exposes neither: the
//! task parks on [`NativeHarnessInner::wake`] while paused, and observes
//! `stopped` / the recorded abort mode at every frame boundary and at the
//! terminal bookkeeping. A turn that ends after `stop` simply emits into the
//! closed channel, which drops the event silently.

use std::sync::Arc;
use std::time::Instant;

use futures::StreamExt;
use openjiuwen_agent::output::{AgentChunk, AgentOutput, ANSWER, LLM_USAGE};
use openjiuwen_agent::Agent;
use openjiuwen_harness_protocol::{
    CheckpointReason, HarnessState, JsonObject, JsonValue, TurnError, TurnEventKind,
    TurnLifecycleEvent, TurnResult, TurnStatus, TurnTermination, TurnTerminationKind, TurnUsage,
    Validate,
};

use super::bridge;
use super::events::{self, TurnTiming};
use super::inner::{EventIds, NativeHarnessInner, PendingInput, TurnSettlement};

/// Spawns the tracked turn task that runs `first` and then every queued input.
pub(crate) fn spawn_turn(
    inner: Arc<NativeHarnessInner>,
    first: PendingInput,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(run_turns(inner, first))
}

/// Runs the accepted turn and then chains queued follow-ups until the queue empties.
///
/// The loop stops as soon as the harness is stopped, aborted, or paused, and
/// settles in [`HarnessState::Idle`] after the last turn of a drained queue.
async fn run_turns(inner: Arc<NativeHarnessInner>, first: PendingInput) {
    let mut next = Some(first);
    while let Some(input) = next {
        if inner.is_stopped() {
            break;
        }
        let aborted = run_turn(&inner, input).await;
        match inner.settle_turn(inner.is_paused(), aborted || inner.is_stopped()) {
            TurnSettlement::Run(queued) => next = Some(queued),
            TurnSettlement::Paused(transition) => {
                if let Some(old) = transition {
                    inner.emit_state_change(old, HarnessState::Paused).await;
                }
                inner.emit_checkpoint(CheckpointReason::StateChanged).await;
                next = None;
            }
            TurnSettlement::Idle(transition) => {
                if let Some(old) = transition {
                    inner.emit_state_change(old, HarnessState::Idle).await;
                }
                inner.emit_checkpoint(CheckpointReason::TurnCompleted).await;
                next = None;
            }
            TurnSettlement::Stopped => next = None,
        }
    }
}

/// Runs one turn and returns whether it was aborted by the host.
async fn run_turn(inner: &Arc<NativeHarnessInner>, input: PendingInput) -> bool {
    let PendingInput {
        turn_id,
        message_id,
        content,
        text: _,
    } = input;
    let old = inner.begin_turn(turn_id.clone(), message_id.clone());
    if old != HarnessState::Running {
        inner.set_state(HarnessState::Running);
        inner.emit_state_change(old, HarnessState::Running).await;
    }
    inner
        .emit(
            TurnLifecycleEvent {
                kind: TurnEventKind::Started,
                result: None,
            }
            .into(),
            EventIds {
                turn_id: Some(&turn_id),
                item_id: None,
                correlation_id: Some(&message_id),
            },
        )
        .await;
    let mut run = TurnRun {
        turn_id,
        message_id,
        started_at: bridge::unix_now(),
        started: Instant::now(),
        content_index: 0,
        output: None,
        usage: None,
        stream_error: None,
    };
    run.drive(inner, content).await;
    run.finish(inner).await
}

/// Bookkeeping of one running turn.
struct TurnRun {
    /// Turn id of this run.
    turn_id: String,
    /// Message id that started this run.
    message_id: String,
    /// Unix epoch seconds when the turn started.
    started_at: f64,
    /// Monotonic start instant used for the duration.
    started: Instant,
    /// Number of output events already emitted by this run.
    content_index: u64,
    /// Terminal kernel output, once the `answer` frame arrived.
    output: Option<AgentOutput>,
    /// Accumulated token usage of this run.
    usage: Option<TurnUsage>,
    /// Kernel stream failure, if any.
    stream_error: Option<String>,
}

impl TurnRun {
    /// Consumes the kernel stream, honouring the pause gate at frame boundaries.
    async fn drive(&mut self, inner: &Arc<NativeHarnessInner>, content: JsonValue) {
        let Some(session) = inner.current_session() else {
            self.stream_error = Some("native turn has no active session".to_owned());
            return;
        };
        {
            let mut stream = inner.agent.stream(content, Some(session));
            while let Some(item) = stream.next().await {
                pause_gate(inner, &self.turn_id, &self.message_id).await;
                if inner.is_stopped() || inner.abort_mode().is_some() {
                    break;
                }
                match item {
                    Err(error) => {
                        self.stream_error = Some(error.to_string());
                        break;
                    }
                    Ok(chunk) => {
                        if self.consume(inner, chunk).await {
                            break;
                        }
                    }
                }
            }
        }
        pause_gate(inner, &self.turn_id, &self.message_id).await;
    }

    /// Publishes one frame and records its answer/usage contribution.
    ///
    /// Returns `true` when the frame ends the stream (an answer that cannot be
    /// decoded), so the caller stops consuming.
    async fn consume(&mut self, inner: &Arc<NativeHarnessInner>, chunk: AgentChunk) -> bool {
        if let Some(payload) = events::chunk_payload(&chunk, &self.turn_id, &mut self.content_index)
        {
            inner
                .emit(
                    payload,
                    EventIds {
                        turn_id: Some(&self.turn_id),
                        item_id: None,
                        correlation_id: None,
                    },
                )
                .await;
        }
        match chunk.kind.as_str() {
            ANSWER => match AgentOutput::from_value(chunk.payload) {
                Ok(output) => self.output = Some(output),
                Err(error) => {
                    self.stream_error = Some(error.to_string());
                    return true;
                }
            },
            LLM_USAGE => {
                if let Some(metadata) = chunk.payload.get("usage_metadata") {
                    if let Some(delta) = events::usage_from_metadata(metadata) {
                        merge_usage(&mut self.usage, delta);
                    }
                }
            }
            _ => {}
        }
        false
    }

    /// Publishes the terminal bookkeeping of the turn and returns whether it was aborted.
    async fn finish(self, inner: &Arc<NativeHarnessInner>) -> bool {
        let TurnRun {
            turn_id,
            message_id,
            started_at,
            started,
            content_index: _,
            output,
            usage,
            stream_error,
        } = self;
        let timing = TurnTiming {
            started_at,
            completed_at: bridge::unix_now(),
            duration_ms: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
        };
        let abort_mode = inner.abort_mode();
        let termination = if abort_mode.is_some() {
            Some(TurnTermination {
                kind: TurnTerminationKind::UserAbort,
                message: Some("native turn was aborted".to_owned()),
                code: None,
                provider_data: JsonObject::new(),
            })
        } else if inner.is_stopped() {
            Some(TurnTermination {
                kind: TurnTerminationKind::HarnessStop,
                message: Some("native harness stopped before the turn completed".to_owned()),
                code: None,
                provider_data: JsonObject::new(),
            })
        } else {
            None
        };
        let interrupted = termination.is_some();
        let mut result = match &stream_error {
            Some(message) => failed_result(message, usage, timing),
            None => events::turn_result(output.as_ref(), usage, timing, termination),
        };
        if let Some(output) = output.as_ref() {
            result.messages = events::messages_from_output(output, &turn_id);
        }
        if let Err(error) = result.validate() {
            tracing::warn!(%error, "native turn result failed validation");
        }
        if !interrupted {
            if let Some(output) = output.as_ref() {
                emit_observations(inner, &turn_id, output).await;
            }
        }
        if let Some(payload) = output.as_ref().and_then(events::failure_diagnostic) {
            inner
                .emit(
                    payload,
                    EventIds {
                        turn_id: Some(&turn_id),
                        item_id: None,
                        correlation_id: None,
                    },
                )
                .await;
        }
        let kind = match result.status {
            TurnStatus::Completed => TurnEventKind::Finished,
            TurnStatus::Interrupted => TurnEventKind::Aborted,
            TurnStatus::Failed => TurnEventKind::Failed,
        };
        inner
            .emit(
                TurnLifecycleEvent {
                    kind,
                    result: Some(result),
                }
                .into(),
                EventIds {
                    turn_id: Some(&turn_id),
                    item_id: None,
                    correlation_id: Some(&message_id),
                },
            )
            .await;
        abort_mode.is_some()
    }
}

/// Emits the coarse `Started` + `Completed` item events of every tool observation.
///
/// [`events::item_events`] returns two payloads per observation in order, so the
/// caller zips the pairs with the observations to attach the tool call id.
async fn emit_observations(inner: &Arc<NativeHarnessInner>, turn_id: &str, output: &AgentOutput) {
    let payloads = events::item_events(output);
    for (pair, observation) in payloads.chunks(2).zip(output.observations.iter()) {
        for payload in pair {
            inner
                .emit(
                    payload.clone(),
                    EventIds {
                        turn_id: Some(turn_id),
                        item_id: observation.tool_call_id.as_deref(),
                        correlation_id: None,
                    },
                )
                .await;
        }
    }
}

/// Blocks the turn while a pause is requested.
///
/// The first entry emits `StateChanged(→ Pausing)` when the turn is still
/// running and a `TurnLifecycle(Paused)` event, then parks on the harness wake
/// signal. A resume emits `Resumed` plus `StateChanged(Paused → Running)`; a
/// stop or abort leaves the turn interrupted without a `Resumed` event.
async fn pause_gate(inner: &Arc<NativeHarnessInner>, turn_id: &str, message_id: &str) {
    let ids = EventIds {
        turn_id: Some(turn_id),
        item_id: None,
        correlation_id: Some(message_id),
    };
    while inner.is_paused() {
        if inner.state_snapshot() == HarnessState::Running {
            inner.set_state(HarnessState::Pausing);
            inner
                .emit_state_change(HarnessState::Running, HarnessState::Pausing)
                .await;
        }
        if inner.set_state(HarnessState::Paused).is_some() {
            inner
                .emit(
                    TurnLifecycleEvent {
                        kind: TurnEventKind::Paused,
                        result: None,
                    }
                    .into(),
                    ids,
                )
                .await;
        }
        loop {
            let notified = inner.wake.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if !inner.is_paused() {
                break;
            }
            if inner.is_stopped() || inner.abort_mode().is_some() {
                break;
            }
            notified.await;
        }
        if inner.is_paused() || inner.is_stopped() || inner.abort_mode().is_some() {
            break;
        }
        inner
            .emit(
                TurnLifecycleEvent {
                    kind: TurnEventKind::Resumed,
                    result: None,
                }
                .into(),
                ids,
            )
            .await;
        inner.set_state(HarnessState::Running);
        inner
            .emit_state_change(HarnessState::Paused, HarnessState::Running)
            .await;
    }
}

/// Builds the `Failed` result of a turn whose kernel stream errored.
fn failed_result(message: &str, usage: Option<TurnUsage>, timing: TurnTiming) -> TurnResult {
    let mut result = events::turn_result(None, usage, timing, None);
    result.status = TurnStatus::Failed;
    result.error = Some(TurnError {
        message: message.to_owned(),
        code: Some("sdk_error".to_owned()),
        category: Some("sdk_error".to_owned()),
        retryable: None,
        provider_data: JsonObject::new(),
    });
    result
}

/// Sums one usage update into the accumulated turn usage.
fn merge_usage(target: &mut Option<TurnUsage>, addition: TurnUsage) {
    let usage = target.get_or_insert_with(TurnUsage::default);
    merge_counter(&mut usage.input_tokens, addition.input_tokens);
    merge_counter(&mut usage.output_tokens, addition.output_tokens);
    merge_counter(&mut usage.cached_input_tokens, addition.cached_input_tokens);
    merge_counter(
        &mut usage.reasoning_output_tokens,
        addition.reasoning_output_tokens,
    );
    merge_counter(&mut usage.total_tokens, addition.total_tokens);
    usage.provider_data.extend(addition.provider_data);
}

/// Adds `value` to `target` when the update reported the counter.
fn merge_counter(target: &mut Option<u64>, value: Option<u64>) {
    if let Some(value) = value {
        *target = Some(target.unwrap_or(0).saturating_add(value));
    }
}
