//! Shared mutable state and lifecycle helpers of the native harness.
//!
//! [`NativeHarnessInner`] is the single copy of the harness state shared by the
//! protocol facade (`mod.rs`), the turn task (`turn.rs`), and the provider. It
//! owns the [`DeepAgent`] handle, the capability card, the bounded observation
//! channel, the checkpoint writer, and the pending/active turn bookkeeping.
//!
//! Every helper keeps its `parking_lot` guard in a tight scope and never holds
//! it across an `await`: [`emit`](NativeHarnessInner::emit) builds the envelope
//! under the lock and awaits the channel afterwards, and
//! [`emit_checkpoint`](NativeHarnessInner::emit_checkpoint) takes the
//! checkpoint writer out of the state before awaiting the host sink.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use openjiuwen_harness_protocol::{
    AbortMode, CheckpointReason, EventBufferConfig, HarnessCard, HarnessCheckpointSink,
    HarnessEventPayload, HarnessState, StateChangedEvent, Validate,
};
use openjiuwen_session::Session;
use parking_lot::Mutex;

use crate::agent::DeepAgent;

use super::bridge::{self, CheckpointWriter, EnvelopeBuilder, NativeCheckpointState};
use super::cursor::EventChannel;
use super::{NativeHarnessOptions, HARNESS_NAME};

/// Correlation identifiers attached to one emitted event.
///
/// All three are optional: lifecycle events carry a turn id and the message id
/// as correlation, output/tool events carry a turn id plus an item id, and pure
/// state changes carry none.
// The field names mirror the protocol envelope id fields, so the `_id` suffix is
// deliberate vocabulary parity rather than a naming accident.
#[allow(clippy::struct_field_names)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct EventIds<'a> {
    /// Turn the event belongs to.
    pub(crate) turn_id: Option<&'a str>,
    /// Provider item the event concerns.
    pub(crate) item_id: Option<&'a str>,
    /// Caller-supplied correlation id (the accepted message id).
    pub(crate) correlation_id: Option<&'a str>,
}

/// Protocol-visible state of one `start`/`stop` cycle.
///
/// The struct starts at [`HarnessState::Terminated`] (an unstarted harness) and
/// is reset by every successful
/// [`start`](openjiuwen_harness_protocol::HarnessProtocol::start).
pub(crate) struct ProtocolState {
    /// Current lifecycle state.
    pub(crate) harness_state: HarnessState,
    /// Agent id of the running cycle.
    pub(crate) agent_id: String,
    /// Host session id of the running cycle.
    pub(crate) host_session_id: String,
    /// Kernel session reused by every turn of the cycle.
    pub(crate) session: Option<Arc<dyn Session>>,
    /// Open observation channel, kept after `stop` until its cursors drain.
    pub(crate) channel: Option<Arc<EventChannel>>,
    /// Envelope builder of the open observation cycle.
    pub(crate) envelopes: Option<EnvelopeBuilder>,
    /// Checkpoint sequence/revision bookkeeping.
    pub(crate) checkpoints: CheckpointWriter,
    /// Host checkpoint sink, when the context supplied one.
    pub(crate) sink: Option<Arc<dyn HarnessCheckpointSink>>,
    /// Number of turns started in the current cycle.
    pub(crate) turn_counter: u64,
    /// Number of messages accepted in the current cycle.
    pub(crate) message_counter: u64,
    /// Kernel iterations completed in the current cycle.
    pub(crate) iteration: u64,
    /// Accepted inputs that have not started a turn yet.
    pub(crate) pending: VecDeque<PendingInput>,
    /// Turn currently driving the kernel stream.
    pub(crate) active_turn: Option<ActiveTurn>,
    /// Join handle of the running turn task.
    pub(crate) task: Option<tokio::task::JoinHandle<()>>,
    /// Model selected for the following turns, when the host chose one.
    pub(crate) model: Option<String>,
    /// Reasoning effort selected for the following turns.
    pub(crate) effort: Option<String>,
}

impl Default for ProtocolState {
    fn default() -> Self {
        Self {
            harness_state: HarnessState::Terminated,
            agent_id: String::new(),
            host_session_id: String::new(),
            session: None,
            channel: None,
            envelopes: None,
            checkpoints: CheckpointWriter::new(HARNESS_NAME),
            sink: None,
            turn_counter: 0,
            message_counter: 0,
            iteration: 0,
            pending: VecDeque::new(),
            active_turn: None,
            task: None,
            model: None,
            effort: None,
        }
    }
}

/// One accepted input waiting for its turn to start.
#[derive(Debug, Clone)]
pub(crate) struct PendingInput {
    /// Turn id reserved for the queued input.
    pub(crate) turn_id: String,
    /// Message id returned to the host when the input was accepted.
    pub(crate) message_id: String,
    /// Kernel input payload handed to `DeepAgent::stream`.
    pub(crate) content: openjiuwen_harness_protocol::JsonValue,
    /// Text projection used for checkpoints.
    pub(crate) text: String,
}

/// The turn currently driving the kernel stream.
#[derive(Debug, Clone)]
pub(crate) struct ActiveTurn {
    /// Turn id of the active turn.
    pub(crate) turn_id: String,
    /// Message id that started the active turn.
    pub(crate) message_id: String,
    /// Abort mode requested while the turn runs, if any.
    pub(crate) abort_mode: Option<AbortMode>,
}

/// What a finished turn leaves for the turn task to do next.
#[derive(Debug)]
pub(crate) enum TurnSettlement {
    /// Run the next queued input.
    Run(PendingInput),
    /// Settle in [`HarnessState::Idle`]; the payload is the previous state when it changed.
    Idle(Option<HarnessState>),
    /// Settle in [`HarnessState::Paused`]; the payload is the previous state when it changed.
    Paused(Option<HarnessState>),
    /// Stop draining: the turn was aborted or the harness is stopping.
    Stopped,
}

/// State and handles shared by every clone of one [`super::NativeHarness`].
///
/// No field is ever mutated through `&mut self`: the protocol surface takes
/// `&self`, so the mutable parts live behind a [`parking_lot::Mutex`] (protocol
/// state) and atomics (pause/stop flags).
pub(crate) struct NativeHarnessInner {
    /// Agent the harness drives.
    pub(crate) agent: Arc<DeepAgent>,
    /// Capability card shared with the provider.
    pub(crate) card: HarnessCard,
    /// Creation options of this harness.
    pub(crate) options: NativeHarnessOptions,
    /// Observation buffer contract advertised through the protocol.
    pub(crate) buffer_config: EventBufferConfig,
    /// Protocol-visible state, never locked across an `await`.
    pub(crate) state: Mutex<ProtocolState>,
    /// Serializes `start`/`stop`/`send`/`resume` spawn decisions.
    pub(crate) start_lock: tokio::sync::Mutex<()>,
    /// Wakes a parked turn task after resume/stop/abort.
    pub(crate) wake: tokio::sync::Notify,
    /// Whether a pause was requested.
    pub(crate) paused: AtomicBool,
    /// Whether the cycle is stopping or stopped.
    pub(crate) stopped: AtomicBool,
    /// Provider session id exposed while a cycle is running.
    ///
    /// The value is written once on the first successful `start` and is never
    /// cleared, so every cycle of this harness reports the same host session id.
    pub(crate) provider_session: OnceLock<String>,
}

impl NativeHarnessInner {
    /// Returns the current lifecycle state.
    pub(crate) fn state_snapshot(&self) -> HarnessState {
        self.state.lock().harness_state
    }

    /// Returns whether a pause was requested.
    pub(crate) fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Acquire)
    }

    /// Returns whether the cycle is stopping or stopped.
    pub(crate) fn is_stopped(&self) -> bool {
        self.stopped.load(Ordering::Acquire)
    }

    /// Returns the abort mode recorded for the active turn.
    pub(crate) fn abort_mode(&self) -> Option<AbortMode> {
        self.state
            .lock()
            .active_turn
            .as_ref()
            .and_then(|turn| turn.abort_mode)
    }

    /// Returns the kernel session of the running cycle.
    pub(crate) fn current_session(&self) -> Option<Arc<dyn Session>> {
        self.state.lock().session.clone()
    }

    /// Returns the agent id of the running cycle.
    pub(crate) fn agent_id(&self) -> String {
        self.state.lock().agent_id.clone()
    }

    /// Returns the host session id of the running cycle.
    pub(crate) fn host_session_id(&self) -> String {
        self.state.lock().host_session_id.clone()
    }

    /// Marks `turn_id` as active and returns the state it replaced.
    pub(crate) fn begin_turn(&self, turn_id: String, message_id: String) -> HarnessState {
        let mut state = self.state.lock();
        state.active_turn = Some(ActiveTurn {
            turn_id,
            message_id,
            abort_mode: None,
        });
        state.harness_state
    }

    /// Settles a finished turn and decides what the turn task does next.
    ///
    /// Clearing the active turn, recording the iteration, and either taking the
    /// next queued input or settling [`HarnessState::Idle`] happen under one
    /// lock, so a concurrent `send` cannot observe a half-settled turn (and
    /// therefore can never spawn a second turn task for it).
    pub(crate) fn settle_turn(&self, paused: bool, halted: bool) -> TurnSettlement {
        let mut state = self.state.lock();
        state.active_turn = None;
        state.iteration = state.iteration.saturating_add(1);
        if paused {
            let old = state.harness_state;
            state.harness_state = HarnessState::Paused;
            return TurnSettlement::Paused((old != HarnessState::Paused).then_some(old));
        }
        if halted {
            return TurnSettlement::Stopped;
        }
        if let Some(input) = state.pending.pop_front() {
            return TurnSettlement::Run(input);
        }
        let old = state.harness_state;
        state.harness_state = HarnessState::Idle;
        TurnSettlement::Idle((old != HarnessState::Idle).then_some(old))
    }

    /// Moves an idle paused harness to [`HarnessState::Idle`] and hands out the next input.
    ///
    /// A running turn keeps the state unchanged and returns no input: the parked
    /// turn task resumes on its own. When no turn is active, the idle transition
    /// and the queue pop happen under one lock so `resume` cannot race `send`.
    pub(crate) fn begin_pending_turn(&self) -> (Option<HarnessState>, Option<PendingInput>) {
        let mut state = self.state.lock();
        if state.active_turn.is_some() {
            return (None, None);
        }
        let old = state.harness_state;
        state.harness_state = HarnessState::Idle;
        let change = (old != HarnessState::Idle).then_some(old);
        (change, state.pending.pop_front())
    }

    /// Stores `new` and returns the previous state when it changed.
    pub(crate) fn set_state(&self, new: HarnessState) -> Option<HarnessState> {
        let mut state = self.state.lock();
        let old = state.harness_state;
        if old == new {
            return None;
        }
        state.harness_state = new;
        Some(old)
    }

    /// Returns the next turn id (`turn-<n>`) from the per-cycle counter.
    pub(crate) fn next_turn_id(&self) -> String {
        bridge::next_turn_id(&mut self.state.lock().turn_counter)
    }

    /// Returns the next message id (`message-<n>`) from the per-cycle counter.
    pub(crate) fn next_message_id(&self) -> String {
        bridge::next_message_id(&mut self.state.lock().message_counter)
    }

    /// Builds the provider checkpoint payload for the current state.
    pub(crate) fn snapshot_checkpoint_state(&self) -> NativeCheckpointState {
        let plan_mode = self.agent.config().default_mode.as_str().to_owned();
        let state = self.state.lock();
        let session_id = state.session.as_ref().map_or_else(
            || state.host_session_id.clone(),
            |session| session.id().to_owned(),
        );
        NativeCheckpointState {
            schema_version: bridge::CHECKPOINT_SCHEMA_VERSION.to_owned(),
            session_id,
            agent_id: state.agent_id.clone(),
            host_session_id: state.host_session_id.clone(),
            iteration: state.iteration,
            pending_follow_ups: state
                .pending
                .iter()
                .map(|input| input.text.clone())
                .collect(),
            plan_mode,
            turn_counter: state.turn_counter,
            model: state.model.clone(),
            effort: state.effort.clone(),
        }
    }

    /// Emits `payload` through the envelope builder and the observation channel.
    ///
    /// A missing envelope builder or channel (an unstarted or fully torn down
    /// harness) is a no-op. The event is validated and a validation failure is
    /// only logged: observation must never fail a turn.
    pub(crate) async fn emit(&self, payload: HarnessEventPayload, ids: EventIds<'_>) {
        let (event, channel) = {
            let mut state = self.state.lock();
            let Some(channel) = state.channel.clone() else {
                return;
            };
            let Some(envelopes) = state.envelopes.as_mut() else {
                return;
            };
            let mut event = envelopes.next(payload);
            if let Some(turn_id) = ids.turn_id {
                event = event.with_turn_id(turn_id);
            }
            if let Some(item_id) = ids.item_id {
                event = event.with_item_id(item_id);
            }
            if let Some(correlation_id) = ids.correlation_id {
                event = event.with_correlation_id(correlation_id);
            }
            (event, channel)
        };
        if let Err(error) = event.validate() {
            tracing::warn!(%error, "native harness event failed validation");
        }
        channel.emit(event).await;
    }

    /// Emits the `old → new` state transition.
    pub(crate) async fn emit_state_change(&self, old: HarnessState, new: HarnessState) {
        self.emit(StateChangedEvent { old, new }.into(), EventIds::default())
            .await;
    }

    /// Publishes a checkpoint for the current state through the host sink.
    ///
    /// The writer is temporarily taken out of the protocol state so the
    /// `parking_lot` guard is never held across the sink `await`; checkpoint
    /// emission is best effort, so a concurrent `export_checkpoint` observing
    /// the taken writer simply reports no checkpoint yet.
    pub(crate) async fn emit_checkpoint(&self, reason: CheckpointReason) {
        let snapshot = self.snapshot_checkpoint_state();
        let (mut writer, sink, provider_session) = {
            let mut state = self.state.lock();
            let writer =
                std::mem::replace(&mut state.checkpoints, CheckpointWriter::new(HARNESS_NAME));
            let sink = state.sink.clone();
            let provider_session = self.provider_session.get().cloned();
            (writer, sink, provider_session)
        };
        let checkpoint = writer
            .emit(
                sink.as_ref(),
                &snapshot,
                provider_session.as_deref(),
                reason,
            )
            .await;
        tracing::debug!(
            sequence = writer.checkpoint_sequence(),
            persisted = checkpoint.is_some(),
            "native harness checkpoint published"
        );
        self.state.lock().checkpoints = writer;
    }
}
