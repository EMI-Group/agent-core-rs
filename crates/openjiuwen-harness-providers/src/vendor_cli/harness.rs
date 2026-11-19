//! The generic harness shell every vendor CLI adapter (and `native_v2`) is wrapped by.
//!
//! [`VendorHarness`] is a complete
//! [`HarnessProtocol`] implementation over one
//! [`VendorDriver`]: the driver owns the vendor wire protocol, the shell owns everything the SPI
//! requires and no vendor should repeat.
//!
//! # What the shell owns
//!
//! - **Lifecycle**: one `start`/`stop` cycle per handle, serialized by an async lock, with the
//!   `Terminated → Idle` transition published as the first observation of the cycle.
//! - **Turns**: [`send`](HarnessProtocol::send) mints `turn-<uuid>` / `message-<uuid>`, queues the
//!   request and returns the receipt immediately. Exactly one supervisor task per harness pops the
//!   queue and runs the turns strictly one after another, so a vendor never sees two concurrent
//!   turns.
//! - **Event sequence**: for each turn the shell publishes `StateChanged(→ Running)`,
//!   `TurnLifecycle(Started)` (correlation id = the accepted message id), the driver's own
//!   observations, one terminal `TurnLifecycle` mapped from the [`TurnStatus`]
//!   (`Completed → Finished`, `Interrupted → Aborted`, `Failed → Failed`), and
//!   `StateChanged(→ Idle)`.
//! - **Failure mapping**: a driver *error* becomes a `Failed` terminal event carrying
//!   [`turn_error`] with the driver's message; a driver *result*
//!   keeps its own normalized [`TurnError`](openjiuwen_harness_protocol::TurnError).
//! - **Capabilities**: steering, both abort modes, pause/resume and checkpoints are gated on the
//!   card; a card that does not declare one produces
//!   [`HarnessError::UnsupportedCapability`] before the driver is consulted.
//! - **Teardown**: [`stop`](HarnessProtocol::stop) closes the driver under a bounded timeout,
//!   drains queued turns as `Started`+`Aborted(HarnessStop)` pairs, publishes
//!   `StateChanged(→ Terminated)`, closes the observation bus and joins the supervisor.
//!
//! # Locking
//!
//! All protocol state lives behind one `std::sync::Mutex` and no guard is ever held across an
//! `.await`: a helper builds values under the lock and publishes them afterwards. `start` and
//! `stop` additionally serialize on a `tokio::sync::Mutex` so a `stop` can never race a `start`
//! into a half-open cycle.
//!
//! # Deviations from the native harness
//!
//! - [`send`](HarnessProtocol::send) with [`DeliveryMode::Auto`](openjiuwen_harness_protocol::DeliveryMode)
//!   always reports `FollowUp`: the supervisor serializes turns, so an accepted input never starts
//!   synchronously.
//! - Aborting with no active turn drains the queue (like the native harness) but ignores the
//!   requested mode, because no turn is left to interrupt.
//! - `resume` settles in `Running` while a turn is still active and in `Idle` otherwise; the
//!   transient `Pausing`/`Resuming` states are never a resting state.
//! - The provider session id reported by
//!   [`provider_session_id`](HarnessProtocol::provider_session_id) is the one of the first
//!   successful `start`; a later cycle that reports a different id is rejected with
//!   [`HarnessError::State`] instead of silently switching identity.
//! - A driver that publishes during [`VendorDriver::open`]
//!   emits before the `Terminated → Idle` transition, which is therefore only guaranteed to be the
//!   first event of a quiet driver.

use std::collections::VecDeque;
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_harness_protocol::{
    AbortMode, DeliveryMode, EventBufferConfig, HarnessCapability, HarnessCard, HarnessCheckpoint,
    HarnessContext, HarnessError, HarnessErrorKind, HarnessEvent, HarnessEventCursor, HarnessInput,
    HarnessProtocol, HarnessResult, HarnessState, SendReceipt, StateChangedEvent, TurnEventKind,
    TurnLifecycleEvent, TurnResult, TurnStatus, TurnTerminationKind, Validate,
};
use uuid::Uuid;

use super::driver::{
    failed_turn, interrupted_turn, turn_error, TurnContext, TurnRequest, VendorDriver,
};
use super::events::{EnvelopeBuilder, EventBus};
use super::lock;

/// Default bound on how long [`start`](HarnessProtocol::start) waits for
/// [`VendorDriver::open`] to complete its handshake.
///
/// Every other wait of the shell is bounded, so a vendor CLI that spawns but never answers must
/// not wedge the handle (and with it the `start_lock` that [`stop`](HarnessProtocol::stop) needs).
/// A harness can override it with [`VendorHarness::with_open_timeout`].
pub const DEFAULT_OPEN_TIMEOUT: Duration = Duration::from_secs(30);

/// Bound on how long `stop` waits for the vendor driver to release its session.
const CLOSE_TIMEOUT: Duration = Duration::from_secs(10);

/// Bound on how long `stop` waits for the turn supervisor to finish.
const SUPERVISOR_TIMEOUT: Duration = Duration::from_secs(5);

/// Protocol-visible state of one `start`/`stop` cycle.
struct VendorState {
    /// Current lifecycle state; `Terminated` before the first `start` and after `stop`.
    harness_state: HarnessState,
    /// Observation bus of the open cycle.
    bus: Option<Arc<EventBus>>,
    /// Envelope builder of the open cycle.
    envelopes: Option<Arc<EnvelopeBuilder>>,
    /// Accepted inputs that have not started a turn yet, oldest first.
    pending: VecDeque<TurnRequest>,
    /// Turn the driver is currently running.
    active: Option<TurnRequest>,
    /// Join handle of the newest supervisor task.
    supervisor: Option<tokio::task::JoinHandle<()>>,
    /// Whether a supervisor task is alive; the guard that keeps exactly one alive.
    running: bool,
    /// Whether the cycle is stopping; the supervisor stops popping turns.
    stopped: bool,
}

impl Default for VendorState {
    fn default() -> Self {
        Self {
            harness_state: HarnessState::Terminated,
            bus: None,
            envelopes: None,
            pending: VecDeque::new(),
            active: None,
            supervisor: None,
            running: false,
            stopped: false,
        }
    }
}

/// Generic [`HarnessProtocol`] shell over a
/// [`VendorDriver`].
///
/// A harness is created **unstarted**: [`state`](HarnessProtocol::state) reports
/// [`HarnessState::Terminated`] until [`start`](HarnessProtocol::start) succeeds. Cloning shares
/// one cycle.
#[derive(Clone)]
pub struct VendorHarness {
    /// Capability card the shell and the driver share.
    card: HarnessCard,
    /// Bounded observation-buffer contract advertised to the host.
    buffer_config: EventBufferConfig,
    /// Vendor half of the harness.
    driver: Arc<dyn VendorDriver>,
    /// Protocol state shared by every clone and by the supervisor task.
    state: Arc<Mutex<VendorState>>,
    /// Serializes `start` and `stop`.
    start_lock: Arc<tokio::sync::Mutex<()>>,
    /// Bound on how long `start` waits for [`VendorDriver::open`] to answer.
    open_timeout: Duration,
    /// Provider session id of the first successful `start`.
    provider_session: Arc<OnceLock<String>>,
}

impl fmt::Debug for VendorHarness {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VendorHarness")
            .field("name", &self.card.name)
            .field("implementation_version", &self.card.implementation_version)
            .field("label", &self.driver.label())
            .field("state", &self.state_snapshot())
            .finish_non_exhaustive()
    }
}

impl VendorHarness {
    /// Wraps `driver` in an **unstarted** shell with `card` and the buffer contract `buffer`.
    ///
    /// `buffer.capacity` sizes the observation bus; `0` selects the bus default. The card is not
    /// validated here — [`start`](HarnessProtocol::start) validates it against the host.
    #[must_use]
    pub fn new(
        card: HarnessCard,
        buffer: EventBufferConfig,
        driver: Arc<dyn VendorDriver>,
    ) -> Self {
        Self {
            card,
            buffer_config: buffer,
            driver,
            state: Arc::new(Mutex::new(VendorState::default())),
            start_lock: Arc::new(tokio::sync::Mutex::new(())),
            open_timeout: DEFAULT_OPEN_TIMEOUT,
            provider_session: Arc::new(OnceLock::new()),
        }
    }

    /// Bounds how long [`start`](HarnessProtocol::start) waits for the driver's `open`.
    ///
    /// Defaults to [`DEFAULT_OPEN_TIMEOUT`]. The timeout is part of the handle, so it applies to
    /// every `start` of this harness; a driver that does not answer within it fails the start with
    /// [`HarnessError::Protocol`], is closed best-effort, and leaves the handle `Terminated` and
    /// reusable.
    #[must_use]
    pub fn with_open_timeout(mut self, timeout: Duration) -> Self {
        self.open_timeout = timeout;
        self
    }

    /// Returns the human label of the wrapped driver, for example `"Claude Code"`.
    #[must_use]
    pub fn label(&self) -> &'static str {
        self.driver.label()
    }

    /// Returns the current lifecycle state without blocking other commands for long.
    fn state_snapshot(&self) -> HarnessState {
        lock(&self.state).harness_state
    }

    /// Builds the observation channel of the open cycle.
    ///
    /// # Errors
    ///
    /// [`HarnessError::State`] when no cycle is open (before `start` or after `stop`).
    fn session(&self) -> HarnessResult<TurnContext> {
        let state = lock(&self.state);
        match (&state.bus, &state.envelopes) {
            (Some(bus), Some(envelopes)) => {
                Ok(TurnContext::new(Arc::clone(bus), Arc::clone(envelopes)))
            }
            _ => Err(HarnessError::state(format!(
                "the {} harness is not started",
                self.driver.label()
            ))),
        }
    }

    /// Records the provider session id `opened` of a new cycle.
    ///
    /// # Errors
    ///
    /// [`HarnessError::State`] when the handle already reports a provider session and the new cycle
    /// does not reproduce it: a vendor session id is the handle's identity, so switching it would
    /// silently re-scope every checkpoint the host stored.
    fn adopt_session(&self, opened: Option<&str>) -> HarnessResult<()> {
        match self.provider_session.get() {
            None => {
                if let Some(id) = opened {
                    let _ = self.provider_session.set(id.to_owned());
                }
                Ok(())
            }
            Some(existing) if opened == Some(existing.as_str()) => Ok(()),
            Some(_) => Err(HarnessError::state(format!(
                "the {} harness cannot be restarted with a different provider session",
                self.driver.label()
            ))),
        }
    }

    /// Closes a driver whose `open` failed or timed out, under the teardown bound.
    ///
    /// The start path owns no cycle yet, so this is best-effort: a close error or a close that
    /// misses [`CLOSE_TIMEOUT`] is logged and never replaces the start error the caller sees. It
    /// runs while `start` still holds the start lock, but it is bounded, so it can never make
    /// `stop` unreachable.
    async fn close_after_failed_open(&self) {
        match tokio::time::timeout(CLOSE_TIMEOUT, self.driver.close()).await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                tracing::warn!(%error, "the vendor driver failed to close after a failed start");
            }
            Err(_) => {
                tracing::warn!(
                    label = self.driver.label(),
                    "the vendor driver did not close after a failed start within the timeout"
                );
            }
        }
    }

    /// Publishes the synthetic `Started` + `Aborted(HarnessStop)` pair of one queued turn.
    async fn abort_queued(&self, session: &TurnContext, turn: &TurnRequest) {
        let started = TurnLifecycleEvent {
            kind: TurnEventKind::Started,
            result: None,
        };
        session
            .emit_correlated(started, Some(&turn.turn_id), Some(&turn.message_id))
            .await;
        let message = format!(
            "the {} harness stopped before the queued turn started",
            self.driver.label()
        );
        let aborted = TurnLifecycleEvent {
            kind: TurnEventKind::Aborted,
            result: Some(interrupted_turn(TurnTerminationKind::HarnessStop, message)),
        };
        session
            .emit_correlated(aborted, Some(&turn.turn_id), Some(&turn.message_id))
            .await;
    }

    /// Drains the queued turns as paired `Started` + `Aborted(HarnessStop)` events.
    async fn drain_queued(&self) {
        let pending: Vec<TurnRequest> = {
            let mut state = lock(&self.state);
            state.pending.drain(..).collect()
        };
        if pending.is_empty() {
            return;
        }
        let Ok(session) = self.session() else {
            return;
        };
        for turn in &pending {
            self.abort_queued(&session, turn).await;
        }
    }

    /// Builds the error cursor reported when no observation cycle is open.
    fn unopened_cursor(message: String) -> Box<dyn HarnessEventCursor> {
        Box::new(ErrorCursor::new(HarnessError::state(message)))
    }

    /// Opens a cursor over the cycle, mapping a missing/closed cycle to an error cursor.
    fn observation_cursor<F>(&self, view: F) -> Box<dyn HarnessEventCursor>
    where
        F: FnOnce(&Arc<EventBus>) -> HarnessResult<Box<dyn HarnessEventCursor>>,
    {
        let bus = { lock(&self.state).bus.clone() };
        let message = format!(
            "the {} harness has no active event cycle",
            self.driver.label()
        );
        match bus {
            Some(bus) if !bus.is_closed() => match view(&bus) {
                Ok(cursor) => cursor,
                Err(error) => Box::new(ErrorCursor::new(error)),
            },
            _ => Self::unopened_cursor(message),
        }
    }
}

#[async_trait]
// The DESIGN-fixed trait signatures take owned inputs; the shell keeps them.
#[allow(clippy::needless_pass_by_value)]
impl HarnessProtocol for VendorHarness {
    fn card(&self) -> &HarnessCard {
        &self.card
    }

    fn state(&self) -> HarnessState {
        self.state_snapshot()
    }

    /// Returns the provider session id, which is only set between start and stop.
    fn provider_session_id(&self) -> Option<&str> {
        if self.state_snapshot() == HarnessState::Terminated {
            return None;
        }
        self.provider_session.get().map(String::as_str)
    }

    fn event_buffer_config(&self) -> &EventBufferConfig {
        &self.buffer_config
    }

    /// Validates the host, opens the driver session, and settles in `Idle`.
    ///
    /// The driver's `open` runs under [`open_timeout`](Self::with_open_timeout)
    /// ([`DEFAULT_OPEN_TIMEOUT`] unless overridden), so a CLI that spawns but never answers fails
    /// the start instead of wedging the handle and the start lock. Every failure path closes the
    /// driver best-effort (bounded) and leaves the harness `Terminated`; the start lock is released
    /// on every exit path, so `stop` and a later `start` stay reachable.
    ///
    /// # Errors
    ///
    /// [`HarnessError::State`] when a cycle already runs or a restart would change the provider
    /// session id, [`HarnessError::Protocol`] for a host/protocol mismatch or when the driver does
    /// not open within the timeout, and the driver's own error unchanged when it cannot open its
    /// session (the observation bus is closed and the harness stays `Terminated`).
    async fn start(&self, context: HarnessContext) -> HarnessResult<()> {
        let _guard = self.start_lock.lock().await;
        if self.state_snapshot() != HarnessState::Terminated {
            return Err(HarnessError::state(format!(
                "{} harness is already started",
                self.driver.label()
            )));
        }
        self.card
            .validate_host(&context.protocol_version, &context.host_capabilities)?;
        let bus = EventBus::new(capacity_of(self.buffer_config.capacity));
        let envelopes = Arc::new(EnvelopeBuilder::new(
            context.host_session_id.clone(),
            context.agent_id.clone(),
        ));
        let session = TurnContext::new(Arc::clone(&bus), Arc::clone(&envelopes));
        let opened =
            match tokio::time::timeout(self.open_timeout, self.driver.open(&context, &session))
                .await
            {
                Ok(Ok(opened)) => opened,
                Ok(Err(error)) => {
                    self.close_after_failed_open().await;
                    bus.close();
                    return Err(error);
                }
                Err(_) => {
                    self.close_after_failed_open().await;
                    bus.close();
                    return Err(HarnessError::protocol(format!(
                        "the {} harness did not open within {}s",
                        self.driver.label(),
                        self.open_timeout.as_secs_f64()
                    )));
                }
            };
        if let Err(error) = self.adopt_session(opened.as_deref()) {
            self.close_after_failed_open().await;
            bus.close();
            return Err(error);
        }
        envelopes.set_provider_session_id(opened);
        {
            let mut state = lock(&self.state);
            state.harness_state = HarnessState::Idle;
            state.bus = Some(Arc::clone(&bus));
            state.envelopes = Some(envelopes);
            state.pending.clear();
            state.active = None;
            state.supervisor = None;
            state.running = false;
            state.stopped = false;
        }
        session
            .emit(
                StateChangedEvent {
                    old: HarnessState::Terminated,
                    new: HarnessState::Idle,
                },
                None,
            )
            .await;
        Ok(())
    }

    /// Closes the driver session, aborts queued turns, and settles in `Terminated`.
    ///
    /// Idempotent: stopping an already terminated harness is a no-op.
    ///
    /// # Errors
    ///
    /// [`HarnessError::State`] when the driver does not close within the teardown timeout. Teardown
    /// still completes, so the harness is `Terminated` even when this error is returned.
    async fn stop(&self) -> HarnessResult<()> {
        let _guard = self.start_lock.lock().await;
        if self.state_snapshot() == HarnessState::Terminated {
            return Ok(());
        }
        lock(&self.state).stopped = true;
        let mut failure = None;
        match tokio::time::timeout(CLOSE_TIMEOUT, self.driver.close()).await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                tracing::warn!(%error, "the vendor driver failed to close cleanly");
                failure = Some(error);
            }
            Err(_) => {
                failure = Some(HarnessError::state(format!(
                    "the {} harness did not stop cleanly",
                    self.driver.label()
                )));
            }
        }
        self.drain_queued().await;
        let session = self.session().ok();
        let old = {
            let mut state = lock(&self.state);
            let old = state.harness_state;
            state.harness_state = HarnessState::Terminated;
            old
        };
        if let Some(session) = session.as_ref() {
            session
                .emit(
                    StateChangedEvent {
                        old,
                        new: HarnessState::Terminated,
                    },
                    None,
                )
                .await;
        }
        let bus = { lock(&self.state).bus.clone() };
        if let Some(bus) = bus {
            bus.close();
        }
        let handle = { lock(&self.state).supervisor.take() };
        if let Some(mut handle) = handle {
            if tokio::time::timeout(SUPERVISOR_TIMEOUT, &mut handle)
                .await
                .is_err()
            {
                tracing::warn!("the vendor turn supervisor did not stop in time; aborting it");
                handle.abort();
            }
        }
        {
            let mut state = lock(&self.state);
            state.harness_state = HarnessState::Terminated;
            state.active = None;
            state.pending.clear();
            state.running = false;
        }
        match failure {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn events(&self) -> Box<dyn HarnessEventCursor> {
        self.observation_cursor(EventBus::events)
    }

    fn turn_events(&self, turn_id: Option<&str>) -> Box<dyn HarnessEventCursor> {
        let turn = turn_id.map(str::to_owned);
        self.observation_cursor(move |bus| bus.turn_events(turn.as_deref()))
    }

    /// Accepts one input and returns its message-to-turn association without waiting for the turn.
    ///
    /// # Errors
    ///
    /// [`HarnessError::State`] when the harness is stopped or, for an explicit steer, when no turn
    /// is active, and [`HarnessError::UnsupportedCapability`] when the card does not declare
    /// steering.
    async fn send(&self, input: HarnessInput, mode: DeliveryMode) -> HarnessResult<SendReceipt> {
        if self.state_snapshot() == HarnessState::Terminated {
            return Err(HarnessError::state(format!(
                "cannot send to a stopped {} harness",
                self.driver.label()
            )));
        }
        if mode == DeliveryMode::Steer {
            return self.send_steer(input).await;
        }
        let turn_id = format!("turn-{}", Uuid::new_v4());
        let message_id = format!("message-{}", Uuid::new_v4());
        let request = TurnRequest::new(
            turn_id.clone(),
            message_id.clone(),
            input.content,
            input.metadata,
        );
        {
            let mut state = lock(&self.state);
            state.pending.push_back(request);
            if !state.running {
                state.running = true;
                state.supervisor = Some(tokio::spawn(supervise(
                    Arc::clone(&self.state),
                    Arc::clone(&self.driver),
                )));
            }
        }
        Ok(SendReceipt {
            message_id,
            turn_id,
            accepted_mode: DeliveryMode::FollowUp,
        })
    }

    /// Aborts the active turn, or every queued turn when none is active.
    ///
    /// # Errors
    ///
    /// [`HarnessError::UnsupportedCapability`] when the card does not declare the requested abort
    /// mode, and the driver's own error when the interrupt cannot be delivered.
    async fn abort(&self, mode: AbortMode) -> HarnessResult<()> {
        if self.state_snapshot() == HarnessState::Terminated {
            return Ok(());
        }
        let active = { lock(&self.state).active.clone() };
        let Some(turn) = active else {
            self.drain_queued().await;
            return Ok(());
        };
        let capability = match mode {
            AbortMode::Force => HarnessCapability::ForceAbort,
            AbortMode::Graceful => HarnessCapability::GracefulAbort,
        };
        if !self.card.supports(capability) {
            return Err(HarnessError::unsupported_capability(format!(
                "the {} harness does not support {} abort",
                self.driver.label(),
                match mode {
                    AbortMode::Force => "force",
                    AbortMode::Graceful => "graceful",
                }
            )));
        }
        tracing::debug!(turn_id = %turn.turn_id, ?mode, "vendor harness abort requested");
        let session = self.session()?;
        self.driver.interrupt(mode, &session).await
    }

    /// Pauses the running turn when `PAUSE_RESUME` is declared.
    ///
    /// # Errors
    ///
    /// [`HarnessError::UnsupportedCapability`] when the card does not declare pause/resume,
    /// [`HarnessError::State`] when the harness has not been started, and the driver's own error
    /// when the pause cannot be delivered.
    async fn pause(&self) -> HarnessResult<()> {
        if !self.card.supports(HarnessCapability::PauseResume) {
            return Err(HarnessError::unsupported_capability(format!(
                "the {} harness cannot pause",
                self.driver.label()
            )));
        }
        let snapshot = self.state_snapshot();
        if snapshot == HarnessState::Terminated {
            return Err(HarnessError::state(format!(
                "the {} harness has not been started",
                self.driver.label()
            )));
        }
        if matches!(snapshot, HarnessState::Paused | HarnessState::Pausing) {
            return Ok(());
        }
        let active = { lock(&self.state).active.clone() };
        let session = self.session()?;
        self.driver.pause(&session).await?;
        if let Some(turn) = active.as_ref() {
            session
                .emit(
                    TurnLifecycleEvent {
                        kind: TurnEventKind::Paused,
                        result: None,
                    },
                    Some(&turn.turn_id),
                )
                .await;
        }
        let old = {
            let mut state = lock(&self.state);
            let old = state.harness_state;
            state.harness_state = HarnessState::Paused;
            old
        };
        session
            .emit(
                StateChangedEvent {
                    old,
                    new: HarnessState::Paused,
                },
                None,
            )
            .await;
        Ok(())
    }

    /// Resumes a paused turn, optionally with `query` as the continuing input.
    ///
    /// # Errors
    ///
    /// [`HarnessError::UnsupportedCapability`] when the card does not declare pause/resume,
    /// [`HarnessError::State`] when the harness is not paused, and the driver's own error when the
    /// resume cannot be delivered.
    async fn resume(&self, query: Option<HarnessInput>) -> HarnessResult<()> {
        if !self.card.supports(HarnessCapability::PauseResume) {
            return Err(HarnessError::unsupported_capability(format!(
                "the {} harness cannot resume",
                self.driver.label()
            )));
        }
        let snapshot = self.state_snapshot();
        if !matches!(snapshot, HarnessState::Paused | HarnessState::Pausing) {
            return Err(HarnessError::state(format!(
                "the {} harness is not paused",
                self.driver.label()
            )));
        }
        let active = { lock(&self.state).active.clone() };
        let session = self.session()?;
        self.driver.resume(query.as_ref(), &session).await?;
        if let Some(turn) = active.as_ref() {
            session
                .emit(
                    TurnLifecycleEvent {
                        kind: TurnEventKind::Resumed,
                        result: None,
                    },
                    Some(&turn.turn_id),
                )
                .await;
        }
        let target = if active.is_some() {
            HarnessState::Running
        } else {
            HarnessState::Idle
        };
        let old = {
            let mut state = lock(&self.state);
            let old = state.harness_state;
            state.harness_state = target;
            old
        };
        session
            .emit(StateChangedEvent { old, new: target }, None)
            .await;
        Ok(())
    }

    /// Returns the driver's provider snapshot.
    ///
    /// # Errors
    ///
    /// [`HarnessError::UnsupportedCapability`] when the card does not declare checkpoints, and the
    /// driver's own error when it cannot serialize its state.
    async fn export_checkpoint(&self) -> HarnessResult<Option<HarnessCheckpoint>> {
        if !self.card.supports(HarnessCapability::Checkpoint) {
            return Err(HarnessError::unsupported_capability(format!(
                "the {} harness does not support checkpoints",
                self.driver.label()
            )));
        }
        self.driver.export_checkpoint().await
    }
}

impl VendorHarness {
    /// Steers `input` into the running turn and returns its receipt.
    ///
    /// # Errors
    ///
    /// [`HarnessError::UnsupportedCapability`] without the steering capability,
    /// [`HarnessError::State`] when no turn is active, and the driver's own error.
    async fn send_steer(&self, input: HarnessInput) -> HarnessResult<SendReceipt> {
        if !self.card.supports(HarnessCapability::Steer) {
            return Err(HarnessError::unsupported_capability(format!(
                "the {} harness does not support steering",
                self.driver.label()
            )));
        }
        let active = { lock(&self.state).active.clone() };
        let Some(turn) = active else {
            return Err(HarnessError::state("there is no active turn to steer"));
        };
        let session = self.session()?;
        self.driver.steer(&turn, &input.content, &session).await?;
        Ok(SendReceipt {
            message_id: format!("message-{}", Uuid::new_v4()),
            turn_id: turn.turn_id,
            accepted_mode: DeliveryMode::Steer,
        })
    }
}

/// Runs the turn queue until it is empty, then clears the supervisor bookkeeping.
///
/// Exactly one instance runs per harness: [`HarnessProtocol::send`] flips `running` before it
/// spawns, and the supervisor resets it under the same mutex, so a send can never start a second
/// one. A stopped harness stops popping turns; whatever is still queued is drained by `stop`.
async fn supervise(state: Arc<Mutex<VendorState>>, driver: Arc<dyn VendorDriver>) {
    while let Some(turn) = pop_next(&state) {
        run_one_turn(&state, driver.as_ref(), turn).await;
    }
    let mut guard = lock(&state);
    guard.running = false;
    guard.supervisor = None;
}

/// Pops the oldest queued turn unless the cycle is stopping.
fn pop_next(state: &Mutex<VendorState>) -> Option<TurnRequest> {
    let mut guard = lock(state);
    if guard.stopped {
        return None;
    }
    guard.pending.pop_front()
}

/// Runs one accepted turn and publishes its lifecycle.
async fn run_one_turn(state: &Mutex<VendorState>, driver: &dyn VendorDriver, turn: TurnRequest) {
    let (session, old) = {
        let mut guard = lock(state);
        guard.active = Some(turn.clone());
        let old = guard.harness_state;
        guard.harness_state = HarnessState::Running;
        let (Some(bus), Some(envelopes)) = (&guard.bus, &guard.envelopes) else {
            return;
        };
        (
            TurnContext::new(Arc::clone(bus), Arc::clone(envelopes)),
            old,
        )
    };
    if old != HarnessState::Running {
        session
            .emit(
                StateChangedEvent {
                    old,
                    new: HarnessState::Running,
                },
                None,
            )
            .await;
    }
    session
        .emit_correlated(
            TurnLifecycleEvent {
                kind: TurnEventKind::Started,
                result: None,
            },
            Some(&turn.turn_id),
            Some(&turn.message_id),
        )
        .await;
    let terminal = match driver.run_turn(&turn, &session).await {
        Ok(result) => {
            if let Err(error) = result.validate() {
                tracing::warn!(%error, "vendor turn result failed validation");
            }
            terminal_event(result)
        }
        Err(error) => {
            tracing::debug!(%error, "vendor driver failed a turn");
            terminal_event(failed_turn(turn_error(
                error.message(),
                error_code(error.kind()),
                "protocol",
                false,
            )))
        }
    };
    session
        .emit_correlated(terminal, Some(&turn.turn_id), Some(&turn.message_id))
        .await;
    settle(state, &session).await;
}

/// Maps a terminal result to its lifecycle event: the status decides the kind.
fn terminal_event(result: TurnResult) -> TurnLifecycleEvent {
    let kind = match result.status {
        TurnStatus::Completed => TurnEventKind::Finished,
        TurnStatus::Interrupted => TurnEventKind::Aborted,
        TurnStatus::Failed => TurnEventKind::Failed,
    };
    TurnLifecycleEvent {
        kind,
        result: Some(result),
    }
}

/// Clears the active turn and settles in `Idle`, unless the cycle is stopping.
async fn settle(state: &Mutex<VendorState>, session: &TurnContext) {
    let transition = {
        let mut guard = lock(state);
        guard.active = None;
        if guard.stopped || guard.harness_state == HarnessState::Idle {
            None
        } else {
            let old = guard.harness_state;
            guard.harness_state = HarnessState::Idle;
            Some(old)
        }
    };
    if let Some(old) = transition {
        session
            .emit(
                StateChangedEvent {
                    old,
                    new: HarnessState::Idle,
                },
                None,
            )
            .await;
    }
}

/// Returns the normalized failure code of a driver error kind.
fn error_code(kind: HarnessErrorKind) -> &'static str {
    match kind {
        HarnessErrorKind::Protocol => "protocol",
        HarnessErrorKind::State => "state",
        HarnessErrorKind::UnsupportedCapability => "unsupported_capability",
        HarnessErrorKind::CheckpointConflict => "checkpoint_conflict",
        HarnessErrorKind::Validation => "validation",
        _ => "harness_error",
    }
}

/// Converts the advertised buffer capacity to the bus capacity (`0` selects the bus default).
fn capacity_of(capacity: u64) -> usize {
    usize::try_from(capacity).unwrap_or(usize::MAX)
}

/// Cursor that reports one [`HarnessError`] and then ends.
///
/// The shell returns it whenever no observation cycle is open or the single-consumer lease is
/// already taken, so a host observes the failure on `next_event` exactly like a provider error.
struct ErrorCursor {
    /// Error to surface on the first read; taken so it is reported only once.
    error: Option<HarnessError>,
}

impl ErrorCursor {
    /// Builds a cursor that reports `error` on its first read.
    fn new(error: HarnessError) -> Self {
        Self { error: Some(error) }
    }
}

#[async_trait]
impl HarnessEventCursor for ErrorCursor {
    async fn next_event(&mut self) -> HarnessResult<Option<HarnessEvent>> {
        match self.error.take() {
            Some(error) => Err(error),
            None => Ok(None),
        }
    }

    async fn close(&mut self) -> HarnessResult<()> {
        self.error = None;
        Ok(())
    }
}
