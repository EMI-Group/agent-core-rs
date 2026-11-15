//! Native harness protocol/provider implementation, event bridge, and cursors.
//!
//! [`NativeHarness`] is the provider-neutral
//! [`HarnessProtocol`] implementation
//! over the kernel [`DeepAgent`]: one harness handle owns one `start`/`stop`
//! cycle, publishes an ordered observation stream, and accepts turns,
//! steering, aborts, pause/resume, checkpoints, and model selection.
//!
//! The module tree:
//!
//! - `mod.rs` — public surface, creation, lifecycle, and model control;
//! - `inner` — shared mutable protocol state and lifecycle helpers;
//! - `turn` — the turn task that drives `DeepAgent::stream`;
//! - `events` — kernel frame/output → protocol payload mapping;
//! - `bridge` — envelopes, id counters, and checkpoint persistence;
//! - `cursor` — the bounded observation channel and its cursors;
//! - `provider` — `NativeHarnessProvider`, the JSON-driven factory.
//!
//! Documented deviations from the Python provider: turn ids are the
//! deterministic `turn-<n>` and message ids `message-<n>` instead of uuid hexes,
//! and `pause_reason`-less pause semantics are harness-level because the kernel
//! exposes no pause API.

use std::collections::BTreeSet;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_foundation::ModelRequestConfig;
use openjiuwen_harness_protocol::{
    AbortMode, CheckpointReason, DeliveryMode, EventBufferConfig, EventOverflowPolicy,
    HarnessCapability, HarnessCard, HarnessCheckpoint, HarnessContext, HarnessError,
    HarnessEventCursor, HarnessInput, HarnessModelControl, HarnessProtocol, HarnessResult,
    HarnessState, HostCapability, JsonObject, JsonValue, ModelOption, ModelSelection, ResumePolicy,
    SendReceipt, TurnEventKind, TurnLifecycleEvent, TurnTermination, TurnTerminationKind, Validate,
};
use openjiuwen_session::{AgentSession, AgentSessionOptions, Session};

use crate::agent::DeepAgent;
use crate::schema::AgentMode;

use bridge::EnvelopeBuilder;
use cursor::EventChannel;
use inner::{EventIds, NativeHarnessInner, PendingInput, ProtocolState};

mod bridge;
mod cursor;
mod events;
mod inner;
mod provider;
mod turn;

#[cfg(test)]
mod tests;

pub use bridge::NativeCheckpointState;
pub use provider::NativeHarnessProvider;

/// Stable identifier of the native harness implementation.
pub const HARNESS_NAME: &str = "openjiuwen-native";

/// Stable identifier of the native harness provider.
pub const PROVIDER_NAME: &str = "native";

/// Version of this crate, reported as the provider implementation version.
pub const PROVIDER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default observation buffer capacity advertised by [`NativeHarnessOptions`].
const DEFAULT_EVENT_BUFFER_CAPACITY: u64 = 1024;

/// Upper bound on how long [`NativeHarness::stop`] waits for the turn task.
const STOP_TIMEOUT: Duration = Duration::from_secs(5);

/// Builds the capability card shared by the native provider and every harness.
///
/// The card declares steering, both abort modes, pause/resume, a persistent
/// session, checkpoints, native and MCP tools, and model discovery/selection.
/// It requires no host capability and can optionally exploit a checkpoint sink,
/// tool approval, user input, and telemetry. `fork_modes` is empty: the native
/// harness does not materialise session forks. The protocol version fields keep
/// the [`HarnessCard::new`] defaults.
#[must_use]
pub fn native_card(
    name: impl Into<String>,
    implementation_version: impl Into<String>,
) -> HarnessCard {
    let mut card = HarnessCard::new(name, implementation_version);
    card.capabilities = BTreeSet::from([
        HarnessCapability::Steer,
        HarnessCapability::GracefulAbort,
        HarnessCapability::ForceAbort,
        HarnessCapability::PauseResume,
        HarnessCapability::PersistentSession,
        HarnessCapability::Checkpoint,
        HarnessCapability::NativeTools,
        HarnessCapability::McpTools,
        HarnessCapability::ModelSelection,
        HarnessCapability::ModelDiscovery,
    ]);
    card.required_host_capabilities = BTreeSet::new();
    card.optional_host_capabilities = BTreeSet::from([
        HostCapability::CheckpointSink,
        HostCapability::ToolApproval,
        HostCapability::UserInput,
        HostCapability::Telemetry,
    ]);
    card.fork_modes = BTreeSet::new();
    card
}

/// Creation options of one native harness.
///
/// The options are pure data so a provider can build a harness from a JSON
/// mapping; they configure the bounded observation channel only.
#[derive(Debug, Clone)]
pub struct NativeHarnessOptions {
    /// Requested observation buffer capacity, in events.
    pub event_buffer_capacity: u64,
    /// Overflow policy applied when the observation buffer is full.
    pub overflow: EventOverflowPolicy,
}

impl Default for NativeHarnessOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl NativeHarnessOptions {
    /// Builds the default options: capacity `1024`, [`EventOverflowPolicy::CoalesceOrBlock`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            event_buffer_capacity: DEFAULT_EVENT_BUFFER_CAPACITY,
            overflow: EventOverflowPolicy::CoalesceOrBlock,
        }
    }

    /// Replaces the requested observation buffer capacity.
    #[must_use]
    pub fn with_event_buffer_capacity(mut self, capacity: u64) -> Self {
        self.event_buffer_capacity = capacity;
        self
    }

    /// Replaces the overflow policy applied when the buffer is full.
    #[must_use]
    pub fn with_overflow(mut self, overflow: EventOverflowPolicy) -> Self {
        self.overflow = overflow;
        self
    }
}

/// Native implementation of the provider-neutral harness protocol.
///
/// Cloning is cheap and shares one state; every `start`/`stop` cycle is
/// serialized internally. A harness is created **unstarted**: [`Self::state`]
/// reports [`HarnessState::Terminated`] until
/// [`start`](HarnessProtocol::start) succeeds.
#[derive(Clone)]
pub struct NativeHarness {
    /// Protocol state shared by every clone.
    inner: Arc<NativeHarnessInner>,
}

impl fmt::Debug for NativeHarness {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeHarness")
            .field("name", &self.inner.card.name)
            .field(
                "implementation_version",
                &self.inner.card.implementation_version,
            )
            .field("options", &self.inner.options)
            .field("state", &self.inner.state_snapshot())
            .finish_non_exhaustive()
    }
}

impl NativeHarness {
    /// Wraps `agent` in an **unstarted** harness with `options`.
    ///
    /// The returned harness reports [`HarnessState::Terminated`] until
    /// [`start`](HarnessProtocol::start) accepts a host context.
    #[must_use]
    pub fn new(agent: Arc<DeepAgent>, options: NativeHarnessOptions) -> Self {
        let buffer_config = EventBufferConfig {
            capacity: options.event_buffer_capacity,
            overflow: options.overflow,
        };
        Self {
            inner: Arc::new(NativeHarnessInner {
                agent,
                card: native_card(HARNESS_NAME, PROVIDER_VERSION),
                options,
                buffer_config,
                state: parking_lot::Mutex::new(ProtocolState::default()),
                start_lock: tokio::sync::Mutex::new(()),
                wake: tokio::sync::Notify::new(),
                paused: AtomicBool::new(false),
                stopped: AtomicBool::new(false),
                provider_session: OnceLock::new(),
            }),
        }
    }

    /// Returns the agent this harness drives.
    #[must_use]
    pub fn agent(&self) -> &Arc<DeepAgent> {
        &self.inner.agent
    }

    /// Returns the creation options of this harness.
    #[must_use]
    pub fn options(&self) -> &NativeHarnessOptions {
        &self.inner.options
    }

    /// Returns whether a cycle is currently started.
    ///
    /// This is the `start`/`stop` bookkeeping only: it is `false` before the
    /// first successful `start` and after `stop`, and `true` while the harness is
    /// idle, running, pausing, or paused.
    #[must_use]
    pub fn started(&self) -> bool {
        self.inner.state_snapshot() != HarnessState::Terminated
    }

    /// Decodes the checkpoint the start context asks the harness to resume from.
    ///
    /// `New` ignores a supplied checkpoint, `RequireResume` demands one, and
    /// `ResumeIfAvailable` falls back to a cold start (with a warning) when the
    /// checkpoint is undecodable or scoped to another agent/host session.
    fn resume_state(context: &HarnessContext) -> HarnessResult<Option<NativeCheckpointState>> {
        match context.resume_policy {
            ResumePolicy::New => Ok(None),
            ResumePolicy::RequireResume => match context.checkpoint.as_ref() {
                Some(checkpoint) => Ok(Some(decode_checkpoint(checkpoint, context)?)),
                None => Err(HarnessError::protocol(
                    "native harness requires a checkpoint to resume",
                )),
            },
            ResumePolicy::ResumeIfAvailable => match context.checkpoint.as_ref() {
                Some(checkpoint) => match decode_checkpoint(checkpoint, context) {
                    Ok(state) => Ok(Some(state)),
                    Err(error) => {
                        tracing::warn!(
                            %error,
                            "ignoring an unusable native checkpoint and starting cold"
                        );
                        Ok(None)
                    }
                },
                None => Ok(None),
            },
        }
    }

    /// Restores the provider bookkeeping recorded in `restored`.
    ///
    /// Queued follow-ups are allocated fresh `turn-<n>`/`message-<n>` ids before
    /// the restored turn counter is applied, so a resumed cycle never reuses an
    /// id while still numbering the restored inputs first. Restored follow-ups
    /// are queued, never auto-run: the next `send` or `resume` starts them.
    fn restore_state(&self, restored: &NativeCheckpointState) {
        let mut state = self.inner.state.lock();
        state.pending.clear();
        for text in &restored.pending_follow_ups {
            let turn_id = bridge::next_turn_id(&mut state.turn_counter);
            let message_id = bridge::next_message_id(&mut state.message_counter);
            state.pending.push_back(PendingInput {
                turn_id,
                message_id,
                content: JsonValue::String(text.clone()),
                text: text.clone(),
            });
        }
        state.turn_counter = state.turn_counter.max(restored.turn_counter);
        state.iteration = restored.iteration;
        state.model.clone_from(&restored.model);
        state.effort.clone_from(&restored.effort);
    }

    /// Emits the synthetic `Started` + `Aborted` pair of a turn stopped before it ran.
    async fn abort_queued_turn(&self, input: &PendingInput) {
        let ids = EventIds {
            turn_id: Some(&input.turn_id),
            item_id: None,
            correlation_id: Some(&input.message_id),
        };
        self.inner
            .emit(
                TurnLifecycleEvent {
                    kind: TurnEventKind::Started,
                    result: None,
                }
                .into(),
                ids,
            )
            .await;
        let now = bridge::unix_now();
        let timing = events::TurnTiming {
            started_at: now,
            completed_at: now,
            duration_ms: 0,
        };
        let termination = TurnTermination {
            kind: TurnTerminationKind::HarnessStop,
            message: Some("native harness stopped before the queued turn started".to_owned()),
            code: None,
            provider_data: JsonObject::new(),
        };
        let mut result = events::turn_result(None, None, timing, Some(termination));
        result.messages.clear();
        self.inner
            .emit(
                TurnLifecycleEvent {
                    kind: TurnEventKind::Aborted,
                    result: Some(result),
                }
                .into(),
                ids,
            )
            .await;
    }

    /// Returns the open channel, or the error cursor of a missing/closed cycle.
    fn observation_cursor<F>(&self, view: F) -> Box<dyn HarnessEventCursor>
    where
        F: FnOnce(&Arc<EventChannel>) -> Box<dyn HarnessEventCursor>,
    {
        let channel = {
            let state = self.inner.state.lock();
            state.channel.clone()
        };
        match channel {
            Some(channel) if !channel.is_closed() => view(&channel),
            _ => cursor::error_cursor("native harness has no active event cycle"),
        }
    }
}

/// Decodes `checkpoint` and verifies that it belongs to the start context.
///
/// # Errors
///
/// [`HarnessError::Protocol`] with the bridge message for a foreign schema
/// (`unsupported native checkpoint version`) or an undecodable payload
/// (`invalid native checkpoint`), and with `native checkpoint scope does not
/// match the context` when the checkpoint was taken for another agent or host
/// session.
fn decode_checkpoint(
    checkpoint: &HarnessCheckpoint,
    context: &HarnessContext,
) -> HarnessResult<NativeCheckpointState> {
    let state = bridge::state_from_checkpoint(checkpoint)
        .map_err(|error| HarnessError::protocol(error.message()))?;
    if state.agent_id != context.agent_id || state.host_session_id != context.host_session_id {
        return Err(HarnessError::protocol(
            "native checkpoint scope does not match the context",
        ));
    }
    Ok(state)
}

/// Returns the text projection of a harness input payload.
///
/// A JSON string is used verbatim, an object contributes its string `query`,
/// and anything else uses its JSON rendering.
fn content_text(content: &JsonValue) -> String {
    match content {
        JsonValue::String(text) => text.clone(),
        JsonValue::Object(object) => object
            .get("query")
            .and_then(JsonValue::as_str)
            .map_or_else(|| content.to_string(), str::to_owned),
        _ => content.to_string(),
    }
}

#[async_trait]
// The DESIGN-fixed trait signatures take owned inputs; implementations keep them.
#[allow(clippy::needless_pass_by_value)]
impl HarnessProtocol for NativeHarness {
    fn card(&self) -> &HarnessCard {
        &self.inner.card
    }

    fn state(&self) -> HarnessState {
        self.inner.state_snapshot()
    }

    /// Returns the provider session id, which is only set between start and stop.
    ///
    /// The id itself is stable for the lifetime of this harness handle: it is the
    /// host session id of the first successful `start`. Restarting the harness
    /// with a different host session id fails with
    /// [`HarnessError::State`](`native harness cannot be restarted with a different host session`)
    /// instead of silently switching sessions.
    fn provider_session_id(&self) -> Option<&str> {
        if self.inner.state_snapshot() == HarnessState::Terminated {
            return None;
        }
        self.inner.provider_session.get().map(String::as_str)
    }

    fn event_buffer_config(&self) -> &EventBufferConfig {
        &self.inner.buffer_config
    }

    /// Validates the host context, starts a cycle, and settles in `Idle`.
    ///
    /// A checkpoint accepted by the resume policy is applied before the
    /// `SessionActivated` checkpoint is published, so that activation checkpoint
    /// already describes the resumed state, and the checkpoint sequence
    /// continues after the resumed checkpoint instead of restarting at 1.
    ///
    /// # Errors
    ///
    /// [`HarnessError::State`] when a cycle already runs or the harness is
    /// restarted with a different host session, [`HarnessError::Protocol`] for a
    /// host/protocol mismatch, an unusable required checkpoint, or a failing
    /// model initialization, and
    /// [`HarnessError::UnsupportedCapability`] when the context mounts MCP
    /// servers (the native harness declares `mcp_tools` but does not mount
    /// context-supplied servers).
    async fn start(&self, context: HarnessContext) -> HarnessResult<()> {
        let _guard = self.inner.start_lock.lock().await;
        if self.inner.state_snapshot() != HarnessState::Terminated {
            return Err(HarnessError::state("native harness is already started"));
        }
        self.inner
            .card
            .validate_host(&context.protocol_version, &context.host_capabilities)?;
        if !context.mcp_servers.is_empty() {
            return Err(HarnessError::unsupported_capability(
                "declare MCP servers in the agent manifest; the native harness does not mount context MCP servers",
            ));
        }
        if let Some(existing) = self.inner.provider_session.get() {
            if existing != &context.host_session_id {
                return Err(HarnessError::state(
                    "native harness cannot be restarted with a different host session",
                ));
            }
        }
        let restored = Self::resume_state(&context)?;
        let session: Arc<dyn Session> = Arc::new(AgentSession::with_options(AgentSessionOptions {
            session_id: Some(context.host_session_id.clone()),
            agent_id: Some(context.agent_id.clone()),
            agent_name: Some(context.agent_name.clone()),
            ..AgentSessionOptions::default()
        }));
        {
            let mut state = self.inner.state.lock();
            state.agent_id.clone_from(&context.agent_id);
            state.host_session_id.clone_from(&context.host_session_id);
            state.session = Some(session);
            state.sink.clone_from(&context.checkpoint_sink);
            state.turn_counter = 0;
            state.message_counter = 0;
            state.iteration = 0;
            state.pending.clear();
            state.active_turn = None;
            state.task = None;
        }
        self.inner.stopped.store(false, Ordering::Release);
        self.inner.paused.store(false, Ordering::Release);
        if self.inner.provider_session.get().is_none() {
            self.inner
                .provider_session
                .set(context.host_session_id.clone())
                .map_err(|_| {
                    HarnessError::state("native harness session id was already initialized")
                })?;
        }
        if !context.system_prompt.is_empty() {
            let mut config = self.inner.agent.config();
            config.system_prompt = Some(context.system_prompt.clone());
            self.inner
                .agent
                .configure(config)
                .map_err(|error| HarnessError::protocol(error.to_string()))?;
        }
        if let Some(restored) = restored.as_ref() {
            let mode = AgentMode::parse(&restored.plan_mode).unwrap_or_default();
            self.inner
                .agent
                .switch_mode(mode)
                .map_err(|error| HarnessError::protocol(error.to_string()))?;
        }
        self.inner
            .agent
            .ensure_initialized()
            .await
            .map_err(|error| HarnessError::protocol(error.to_string()))?;
        let channel = EventChannel::open(self.inner.buffer_config);
        let effective = channel.config();
        tracing::debug!(
            capacity = effective.capacity,
            overflow = ?effective.overflow,
            "native harness observation channel opened"
        );
        let mut envelopes =
            EnvelopeBuilder::new(context.host_session_id.clone(), context.agent_id.clone());
        envelopes.set_provider_session_id(Some(context.host_session_id.clone()));
        {
            let mut state = self.inner.state.lock();
            state.channel = Some(channel);
            state.envelopes = Some(envelopes);
            state.harness_state = HarnessState::Idle;
        }
        self.inner
            .emit_state_change(HarnessState::Terminated, HarnessState::Idle)
            .await;
        if let Some(restored) = restored {
            self.restore_state(&restored);
            // The checkpoint sequence is monotonic per provider/agent/host
            // session, so a resumed cycle continues after the checkpoint it was
            // restored from instead of restarting at 1.
            if let Some(checkpoint) = context.checkpoint.as_ref() {
                let mut state = self.inner.state.lock();
                let current = state.checkpoints.checkpoint_sequence();
                state.checkpoints.reset(current.max(checkpoint.sequence));
            }
        }
        self.inner
            .emit_checkpoint(CheckpointReason::SessionActivated)
            .await;
        Ok(())
    }

    /// Stops the cycle, aborts queued turns, and settles in `Terminated`.
    ///
    /// Idempotent: stopping an already terminated harness is a no-op. A cursor
    /// obtained earlier drains the still-buffered events before it ends because
    /// the channel is kept alive (closed) after the cycle.
    ///
    /// # Errors
    ///
    /// Currently infallible; the signature mirrors the protocol trait.
    async fn stop(&self) -> HarnessResult<()> {
        let _guard = self.inner.start_lock.lock().await;
        if self.inner.state_snapshot() == HarnessState::Terminated {
            return Ok(());
        }
        tracing::debug!(
            agent_id = %self.inner.agent_id(),
            host_session_id = %self.inner.host_session_id(),
            "native harness cycle stopping"
        );
        self.inner.stopped.store(true, Ordering::Release);
        self.inner.paused.store(false, Ordering::Release);
        if let Err(error) = self.inner.agent.stop() {
            tracing::warn!(%error, "failed to stop the native harness agent");
        }
        self.inner.wake.notify_waiters();
        let pending: Vec<PendingInput> = {
            let mut state = self.inner.state.lock();
            state.pending.drain(..).collect()
        };
        for input in pending {
            self.abort_queued_turn(&input).await;
        }
        if let Some(old) = self.inner.set_state(HarnessState::Terminated) {
            self.inner
                .emit_state_change(old, HarnessState::Terminated)
                .await;
        }
        let channel = {
            let state = self.inner.state.lock();
            state.channel.clone()
        };
        if let Some(channel) = channel {
            channel.close();
        }
        let handle = {
            let mut state = self.inner.state.lock();
            state.task.take()
        };
        if let Some(mut handle) = handle {
            if tokio::time::timeout(STOP_TIMEOUT, &mut handle)
                .await
                .is_err()
            {
                tracing::warn!("native harness turn task did not stop in time; aborting it");
                handle.abort();
            }
        }
        {
            let mut state = self.inner.state.lock();
            state.harness_state = HarnessState::Terminated;
            state.session = None;
            state.active_turn = None;
            state.pending.clear();
            state.task = None;
            state.sink = None;
        }
        Ok(())
    }

    fn events(&self) -> Box<dyn HarnessEventCursor> {
        self.observation_cursor(EventChannel::events)
    }

    fn turn_events(&self, turn_id: Option<&str>) -> Box<dyn HarnessEventCursor> {
        self.observation_cursor(|channel| channel.turn_events(turn_id))
    }

    /// Accepts one input and returns its message-to-turn association.
    ///
    /// # Errors
    ///
    /// [`HarnessError::State`] when the harness is stopped or (for an explicit
    /// steer) when no turn runs, and
    /// [`HarnessError::UnsupportedCapability`] when the card does not declare
    /// steering.
    async fn send(&self, input: HarnessInput, mode: DeliveryMode) -> HarnessResult<SendReceipt> {
        if self.inner.state_snapshot() == HarnessState::Terminated {
            return Err(HarnessError::state(
                "cannot send to a stopped native harness",
            ));
        }
        let text = content_text(&input.content);
        if mode == DeliveryMode::Steer {
            if !self.inner.card.supports(HarnessCapability::Steer) {
                return Err(HarnessError::unsupported_capability(
                    "openjiuwen-native does not support steering",
                ));
            }
            let turn_id = {
                let state = self.inner.state.lock();
                state.active_turn.as_ref().map(|turn| turn.turn_id.clone())
            };
            let Some(turn_id) = turn_id else {
                return Err(HarnessError::state("there is no active turn to steer"));
            };
            self.inner.agent.steer(text);
            return Ok(SendReceipt {
                message_id: self.inner.next_message_id(),
                turn_id,
                accepted_mode: DeliveryMode::Steer,
            });
        }
        let _guard = self.inner.start_lock.lock().await;
        if self.inner.state_snapshot() == HarnessState::Terminated {
            return Err(HarnessError::state(
                "cannot send to a stopped native harness",
            ));
        }
        let turn_id = self.inner.next_turn_id();
        let message_id = self.inner.next_message_id();
        let queued = PendingInput {
            turn_id: turn_id.clone(),
            message_id: message_id.clone(),
            content: input.content,
            text,
        };
        let accepted_mode = {
            let mut state = self.inner.state.lock();
            let defer = state.active_turn.is_some()
                || self.inner.is_paused()
                || matches!(
                    state.harness_state,
                    HarnessState::Paused | HarnessState::Pausing
                );
            if defer {
                state.pending.push_back(queued);
                DeliveryMode::FollowUp
            } else if let Some(restored) = state.pending.pop_front() {
                // Restored follow-ups never auto-run: the next accepted input
                // starts the queue, and this input runs behind those.
                state.pending.push_back(queued);
                state.task = Some(turn::spawn_turn(Arc::clone(&self.inner), restored));
                DeliveryMode::FollowUp
            } else {
                state.task = Some(turn::spawn_turn(Arc::clone(&self.inner), queued));
                mode
            }
        };
        Ok(SendReceipt {
            message_id,
            turn_id,
            accepted_mode,
        })
    }

    /// Aborts the active turn in `mode`.
    ///
    /// # Errors
    ///
    /// [`HarnessError::UnsupportedCapability`] when the card does not declare
    /// the requested abort mode. Aborting without an active turn is a no-op.
    async fn abort(&self, mode: AbortMode) -> HarnessResult<()> {
        let capability = match mode {
            AbortMode::Force => HarnessCapability::ForceAbort,
            AbortMode::Graceful => HarnessCapability::GracefulAbort,
        };
        if !self.inner.card.supports(capability) {
            return Err(HarnessError::unsupported_capability(match mode {
                AbortMode::Force => "openjiuwen-native does not support force abort",
                AbortMode::Graceful => "openjiuwen-native does not support graceful abort",
            }));
        }
        let active = {
            let mut state = self.inner.state.lock();
            match state.active_turn.as_mut() {
                Some(turn) => {
                    tracing::debug!(
                        turn_id = %turn.turn_id,
                        message_id = %turn.message_id,
                        ?mode,
                        "native harness abort requested"
                    );
                    turn.abort_mode = Some(mode);
                    true
                }
                None => false,
            }
        };
        if !active {
            return Ok(());
        }
        self.inner.agent.abort();
        self.inner.wake.notify_waiters();
        Ok(())
    }

    /// Pauses the harness or the active turn.
    ///
    /// Serialized with `start`/`stop`, so a pause racing a stop can never
    /// resurrect a terminated cycle.
    ///
    /// # Errors
    ///
    /// [`HarnessError::State`] when the harness has not been started.
    async fn pause(&self) -> HarnessResult<()> {
        let _guard = self.inner.start_lock.lock().await;
        let snapshot = self.inner.state_snapshot();
        if snapshot == HarnessState::Terminated {
            return Err(HarnessError::state("native harness is not started"));
        }
        if matches!(snapshot, HarnessState::Paused | HarnessState::Pausing) {
            return Ok(());
        }
        self.inner.paused.store(true, Ordering::Release);
        let active = {
            let state = self.inner.state.lock();
            state.active_turn.is_some()
        };
        if active {
            if let Some(old) = self.inner.set_state(HarnessState::Pausing) {
                self.inner
                    .emit_state_change(old, HarnessState::Pausing)
                    .await;
            }
            return Ok(());
        }
        if let Some(old) = self.inner.set_state(HarnessState::Paused) {
            self.inner
                .emit_state_change(old, HarnessState::Paused)
                .await;
        }
        self.inner
            .emit_checkpoint(CheckpointReason::StateChanged)
            .await;
        Ok(())
    }

    /// Resumes a paused harness, optionally with `query` as the next input.
    ///
    /// # Errors
    ///
    /// [`HarnessError::State`] when the harness is not paused.
    async fn resume(&self, query: Option<HarnessInput>) -> HarnessResult<()> {
        let snapshot = self.inner.state_snapshot();
        if !matches!(snapshot, HarnessState::Paused | HarnessState::Pausing) {
            return Err(HarnessError::state("native harness is not paused"));
        }
        let _guard = self.inner.start_lock.lock().await;
        if !matches!(
            self.inner.state_snapshot(),
            HarnessState::Paused | HarnessState::Pausing
        ) {
            return Err(HarnessError::state("native harness is not paused"));
        }
        if let Some(input) = query {
            let text = content_text(&input.content);
            let mut state = self.inner.state.lock();
            let turn_id = bridge::next_turn_id(&mut state.turn_counter);
            let message_id = bridge::next_message_id(&mut state.message_counter);
            state.pending.push_back(PendingInput {
                turn_id,
                message_id,
                content: input.content,
                text,
            });
        }
        self.inner.paused.store(false, Ordering::Release);
        self.inner.wake.notify_waiters();
        let (transition, next) = self.inner.begin_pending_turn();
        if let Some(old) = transition {
            self.inner.emit_state_change(old, HarnessState::Idle).await;
        }
        if let Some(input) = next {
            let handle = turn::spawn_turn(Arc::clone(&self.inner), input);
            self.inner.state.lock().task = Some(handle);
        }
        Ok(())
    }

    /// Returns the latest locally built checkpoint.
    ///
    /// # Errors
    ///
    /// Currently infallible; the signature mirrors the protocol trait.
    async fn export_checkpoint(&self) -> HarnessResult<Option<HarnessCheckpoint>> {
        let checkpoint = {
            let state = self.inner.state.lock();
            state.checkpoints.latest().cloned()
        };
        Ok(checkpoint)
    }
}

#[async_trait]
// The DESIGN-fixed trait signature takes an owned selection; implementations keep it.
#[allow(clippy::needless_pass_by_value)]
impl HarnessModelControl for NativeHarness {
    /// Reports the configured model as the only catalog entry.
    ///
    /// # Errors
    ///
    /// [`HarnessError::UnsupportedCapability`] when the card does not declare
    /// model discovery.
    async fn list_models(&self) -> HarnessResult<Vec<ModelOption>> {
        if !self.inner.card.supports(HarnessCapability::ModelDiscovery) {
            return Err(HarnessError::unsupported_capability(
                "openjiuwen-native does not support model discovery",
            ));
        }
        let selected = {
            let state = self.inner.state.lock();
            state.model.clone()
        };
        let Some(name) = selected.or_else(|| self.inner.agent.config().model_name) else {
            return Ok(Vec::new());
        };
        Ok(vec![ModelOption {
            model_id: name.clone(),
            display_name: name,
            description: String::new(),
            efforts: vec!["low".to_owned(), "medium".to_owned(), "high".to_owned()],
            default_effort: None,
            is_default: true,
            extensions: JsonObject::new(),
        }])
    }

    /// Switches the model and/or reasoning effort of the following turns.
    ///
    /// # Errors
    ///
    /// [`HarnessError::UnsupportedCapability`] when the card does not declare
    /// model selection, [`HarnessError::State`] before the first `start`,
    /// [`HarnessError::Validation`] for an invalid selection or an unrecognized
    /// reasoning effort.
    async fn set_model(&self, selection: ModelSelection) -> HarnessResult<()> {
        if !self.inner.card.supports(HarnessCapability::ModelSelection) {
            return Err(HarnessError::unsupported_capability(
                "openjiuwen-native does not support model selection",
            ));
        }
        if self.inner.state_snapshot() == HarnessState::Terminated {
            return Err(HarnessError::state("native harness has not been started"));
        }
        selection.validate()?;
        let current = {
            let state = self.inner.state.lock();
            state.model.clone()
        }
        .or_else(|| self.inner.agent.config().model_name);
        let model = match &selection.model {
            Some(name) => Some(name.clone()),
            None => current,
        };
        let temperature: Option<f32> = match selection.effort.as_deref() {
            Some("low") => Some(0.2),
            Some("medium") => Some(0.6),
            Some("high") => Some(1.0),
            Some(effort) => {
                return Err(HarnessError::validation(format!(
                    "unsupported reasoning effort '{effort}'"
                )));
            }
            None => None,
        };
        if selection.model.is_some() || temperature.is_some() {
            let mut config = self.inner.agent.config();
            if let Some(name) = &selection.model {
                config.model_name = Some(name.clone());
            }
            if let Some(temperature) = temperature {
                if let Some(model_name) = &model {
                    let mut request = config
                        .model_config
                        .clone()
                        .unwrap_or_else(|| ModelRequestConfig::new(model_name.clone()));
                    request.model_name.clone_from(model_name);
                    request.temperature = Some(temperature);
                    config.model_config = Some(request);
                }
            }
            self.inner
                .agent
                .configure(config)
                .map_err(|error| HarnessError::protocol(error.to_string()))?;
        }
        {
            let mut state = self.inner.state.lock();
            if selection.model.is_some() {
                state.model = selection.model;
            }
            if selection.effort.is_some() {
                state.effort = selection.effort;
            }
        }
        Ok(())
    }
}
