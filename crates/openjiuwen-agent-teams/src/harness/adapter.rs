//! The adapter from a member's harness protocol to the team's member-runtime seam.
//!
//! Python parity source: `openjiuwen/agent_teams/harness/protocol_adapter.py` and
//! `openjiuwen/agent_teams/harness/native_harness.py` (`send`/`abort`/`pause`/`resume` surface and
//! the `on_state`/`on_round` subscription). Python adapts `NativeHarness` *down* to the
//! provider-neutral protocol; in this port `openjiuwen_harness::NativeHarness` already implements
//! [`HarnessProtocol`], so the adapter only has to *drive* protocol handles and forward their
//! observations to [`MemberObserver`]s.
//!
//! ## Pump ownership
//!
//! [`HarnessMemberRuntime::start`] opens the cycle, publishes the settled state, and owns one
//! background *pump* task that drains `harness.events()` into the registered observers. The pump
//! is owned by a [`PumpHandle`]: it ends when the cancellation token fires (`stop` or drop of the
//! runtime), when the cursor is exhausted, on a cursor error (logged), or when the harness reports
//! [`HarnessState::Terminated`]. The cursor itself is the cycle-long single-consumer view, so a
//! finished turn does not end the pump.
use super::mapping::{chunk_from_event, round_from_event, state_from_event};
use crate::member::{MemberObserver, MemberRuntime};
use crate::schema::MemberMode;
use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_harness_protocol::{
    AbortMode, DeliveryMode, HarnessContext, HarnessError, HarnessEvent, HarnessInput,
    HarnessProtocol, HarnessState,
};
use openjiuwen_session::Session;
use parking_lot::{Mutex, RwLock};
use std::sync::Arc;
use std::time::Duration;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};
/// Bound on how long `stop` waits for the pump task to finish before abandoning it.
const PUMP_STOP_TIMEOUT: Duration = Duration::from_secs(5);
/// Static context a [`MemberHarnessFactory`] needs to materialize one member's harness.
///
/// The factory receives the member's cold [`DeepAgentSpec`](crate::schema::DeepAgentSpec) plus this
/// runtime context because parts of the identity (team, member, session, workspace, language,
/// mode) live in the team runtime rather than in the spec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemberHarnessContext {
    /// Name of the team the member belongs to.
    pub team_name: String,
    /// Name of the member being built.
    pub member_name: String,
    /// Host session identifier the member's harness runs under.
    pub session_id: String,
    /// Root directory the member's tools and workspace files resolve against.
    pub workspace_root: std::path::PathBuf,
    /// Effective prompt language (`cn`, `en`, …).
    pub language: String,
    /// Tool/prompt set the member runs with (build or plan).
    pub mode: MemberMode,
}
impl MemberHarnessContext {
    /// Builds the context from the six runtime identity values.
    #[must_use]
    pub fn new(
        team_name: impl Into<String>,
        member_name: impl Into<String>,
        session_id: impl Into<String>,
        workspace_root: impl Into<std::path::PathBuf>,
        language: impl Into<String>,
        mode: MemberMode,
    ) -> Self {
        Self {
            team_name: team_name.into(),
            member_name: member_name.into(),
            session_id: session_id.into(),
            workspace_root: workspace_root.into(),
            language: language.into(),
            mode,
        }
    }
}
/// Builds one member's harness from its cold spec and runtime context.
///
/// The factory is the only place that knows *which* harness implementation a member runs on:
/// [`super::NativeMemberFactory`] builds the in-process `NativeHarness`, and a deferred
/// external-CLI mode would implement this trait in its own module. The returned harness is
/// **unstarted**; [`HarnessMemberRuntime`] starts it.
#[async_trait]
pub trait MemberHarnessFactory: Send + Sync {
    /// Materializes one unstarted harness for `spec`.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Config`] when a required dependency (model,
    /// workspace, provider) is missing, and the provider's own error when construction fails.
    async fn create(
        &self,
        spec: &crate::schema::DeepAgentSpec,
        ctx: &MemberHarnessContext,
    ) -> Result<Arc<dyn HarnessProtocol>>;
    /// Returns the static factory identity (`"native"` for the in-process factory).
    fn name(&self) -> &'static str;
}
/// Drives one [`HarnessProtocol`] handle through the team's [`MemberRuntime`] seam.
///
/// The runtime owns no agent itself: it forwards member commands to the protocol handle, mirrors
/// the harness state, and pumps the observation stream into subscribed [`MemberObserver`]s.
pub struct HarnessMemberRuntime {
    /// The protocol handle this member drives.
    harness: Arc<dyn HarnessProtocol>,
    /// Host-side session identifier of this member.
    session_id: String,
    /// Registered observers; cheap to clone so the pump never holds the lock across a callback.
    observers: Arc<RwLock<Vec<Arc<dyn MemberObserver>>>>,
    /// The single background pump, when a cycle is running.
    pump: Mutex<Option<PumpHandle>>,
}
impl HarnessMemberRuntime {
    /// Wraps `harness` as the member runtime of `session_id`.
    #[must_use]
    pub fn new(harness: Arc<dyn HarnessProtocol>, session_id: impl Into<String>) -> Self {
        Self {
            harness,
            session_id: session_id.into(),
            observers: Arc::new(RwLock::new(Vec::new())),
            pump: Mutex::new(None),
        }
    }
    /// Returns the protocol handle, for callers that need provider-specific controls.
    #[must_use]
    pub fn harness(&self) -> &Arc<dyn HarnessProtocol> {
        &self.harness
    }
    /// Publishes the harness's current state to every observer, in registration order.
    fn publish_state(&self) {
        let state = self.harness.state();
        let observers: Vec<Arc<dyn MemberObserver>> = self.observers.read().clone();
        for observer in &observers {
            observer.on_state(state);
        }
    }
    /// Opens the observation pump for the cycle `start` just opened.
    fn start_pump(&self) {
        let mut cursor = self.harness.events();
        let observers = Arc::clone(&self.observers);
        let harness = Arc::clone(&self.harness);
        let token = CancellationToken::new();
        let pump_token = token.clone();
        let task = tokio::spawn(async move {
            loop {
                tokio::select! {
                    biased;
                    () = pump_token.cancelled() => break,
                    next = cursor.next_event() => match next {
                        Ok(Some(event)) => {
                            dispatch_event(&observers, &event);
                            if harness.state() == HarnessState::Terminated {
                                break;
                            }
                        }
                        Ok(None) => {
                            if let Err(error) = cursor.close().await {
                                debug!(error = %error, "member harness cursor closed with an error");
                            }
                            break;
                        }
                        Err(error) => {
                            warn!(error = %error, "member harness event pump stopped on a cursor error");
                            break;
                        }
                    }
                }
            }
        });
        *self.pump.lock() = Some(PumpHandle {
            token,
            task: Some(task),
        });
    }
    /// Cancels and joins the pump, if one is running.
    async fn shutdown_pump(&self) {
        let handle = { self.pump.lock().take() };
        if let Some(handle) = handle {
            handle.shutdown().await;
        }
    }
}
#[async_trait]
#[allow(clippy::needless_pass_by_value)] // the seam takes owned content/query strings
impl MemberRuntime for HarnessMemberRuntime {
    /// Returns the host-side session identifier of this member.
    fn session_id(&self) -> &str {
        &self.session_id
    }
    /// Returns the harness lifecycle state verbatim.
    fn state(&self) -> HarnessState {
        self.harness.state()
    }
    /// Starts the harness cycle and its observation pump.
    ///
    /// The protocol [`HarnessContext`] is built from the runtime's own session id; when a team
    /// session is supplied, its id becomes `host_session_id` so the harness can correlate with the
    /// team's session. The member's system prompt and workspace are part of the constructed agent,
    /// not of the start context, so `system_prompt` is empty here.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] for an empty session id and
    /// [`openjiuwen_common::error::ErrorKind::External`] wrapping the protocol's own error.
    async fn start(&self, team_session: Option<Arc<dyn Session>>) -> Result<()> {
        if self.session_id.is_empty() {
            return Err(Error::validation(
                "member harness session id must not be empty",
            ));
        }
        self.shutdown_pump().await;
        let host_session_id = team_session.as_ref().map_or_else(
            || self.session_id.clone(),
            |session| session.id().to_owned(),
        );
        let context = HarnessContext::new(
            self.session_id.clone(),
            self.session_id.clone(),
            host_session_id,
            String::new(),
        );
        self.harness.start(context).await.map_err(harness_error)?;
        self.publish_state();
        self.start_pump();
        Ok(())
    }
    /// Cancels the pump, then stops the harness cycle.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::External`] when the provider cannot close
    /// its cycle cleanly.
    async fn stop(&self) -> Result<()> {
        self.shutdown_pump().await;
        let result = self.harness.stop().await.map_err(harness_error);
        self.publish_state();
        result
    }
    /// Sends member input, steering the active turn when `immediate` is set.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::External`] when the harness rejects the
    /// input (no cycle running, unsupported delivery mode, …).
    async fn send(&self, content: String, immediate: bool) -> Result<()> {
        let mode = if immediate {
            DeliveryMode::Steer
        } else {
            DeliveryMode::FollowUp
        };
        self.harness
            .send(HarnessInput::text(content), mode)
            .await
            .map(|_receipt| ())
            .map_err(harness_error)
    }
    /// Aborts the active turn immediately (`immediate`) or at its next safe point.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::External`] when no turn is active or the
    /// provider does not declare the requested abort capability.
    async fn abort(&self, immediate: bool) -> Result<()> {
        let mode = if immediate {
            AbortMode::Force
        } else {
            AbortMode::Graceful
        };
        self.harness.abort(mode).await.map_err(harness_error)
    }
    /// Pauses the active turn.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::External`] when the harness cannot pause.
    async fn pause(&self) -> Result<()> {
        self.harness.pause().await.map_err(harness_error)
    }
    /// Resumes a paused turn, optionally with the continuation query.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::External`] when no turn is paused or the
    /// provider does not support pause/resume.
    async fn resume(&self, query: Option<String>) -> Result<()> {
        let query = query.map(HarnessInput::text);
        self.harness.resume(query).await.map_err(harness_error)
    }
    /// Registers an observer for state, round and chunk callbacks.
    fn subscribe(&self, observer: Arc<dyn MemberObserver>) {
        self.observers.write().push(observer);
    }
}
/// Translates a protocol error into the crate-wide error type.
///
/// Every protocol failure becomes [`Error::External`] with the [`HarnessError`] kept as the source,
/// so the variant and its message stay reachable through `std::error::Error::source`. The
/// protocol has no timeout variant, so there is no timeout special case.
fn harness_error(error: HarnessError) -> Error {
    Error::External(Box::new(error))
}
/// Forwards every team-relevant projection of one observation, in event order.
fn dispatch_event(observers: &RwLock<Vec<Arc<dyn MemberObserver>>>, event: &HarnessEvent) {
    let state = state_from_event(event);
    let round = round_from_event(event);
    let chunk = chunk_from_event(event);
    if state.is_none() && round.is_none() && chunk.is_none() {
        return;
    }
    let subscribers: Vec<Arc<dyn MemberObserver>> = observers.read().clone();
    if let Some(state) = state {
        for observer in &subscribers {
            observer.on_state(state);
        }
    }
    if let Some(round) = round {
        for observer in &subscribers {
            observer.on_round(round);
        }
    }
    if let Some(chunk) = chunk {
        for observer in &subscribers {
            observer.on_chunk(chunk.clone());
        }
    }
}
/// Owned handle of the observation pump.
///
/// Dropping the handle cancels the pump and aborts its task, which is how a dropped
/// [`HarnessMemberRuntime`] releases its background work.
struct PumpHandle {
    /// Cancellation signal observed by the pump loop.
    token: CancellationToken,
    /// The pump task; taken out by [`PumpHandle::shutdown`] so the handle can still be `Drop`.
    task: Option<JoinHandle<()>>,
}
impl PumpHandle {
    /// Cancels the pump and waits (bounded) for the task to end.
    async fn shutdown(mut self) {
        self.token.cancel();
        if let Some(task) = self.task.take() {
            let _ = tokio::time::timeout(PUMP_STOP_TIMEOUT, task).await;
        }
    }
}
impl Drop for PumpHandle {
    fn drop(&mut self) {
        self.token.cancel();
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}
/// One recorded command the fake harness received.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
enum Call {
    /// `start` was called with the given host session id.
    Start { host_session_id: String },
    /// `stop` was called.
    Stop,
    /// `send` was called with the given delivery mode.
    Send { mode: DeliveryMode },
    /// `abort` was called with the given abort mode.
    Abort { mode: AbortMode },
    /// `pause` was called.
    Pause,
    /// `resume` was called with (`true`) or without a query.
    Resume { has_query: bool },
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::schema::RoundOutcome;
    use openjiuwen_harness_protocol::{
        EventBufferConfig, HarnessCard, HarnessError, HarnessEventCursor, HarnessResult,
        HarnessState, OutputChannel, OutputEvent, OutputKind, OutputOperation, SendReceipt,
        StateChangedEvent, TurnEventKind, TurnLifecycleEvent, VecEventCursor,
    };
    use serde_json::json;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use tokio::sync::mpsc;
    /// Fake protocol handle: records commands, serves a channel-backed observation cursor, and
    /// returns a fixed state.
    struct FakeHarness {
        /// State returned by [`HarnessProtocol::state`].
        state: Mutex<HarnessState>,
        /// Commands received, in order.
        calls: Arc<Mutex<Vec<Call>>>,
        /// Static capability card.
        card: HarnessCard,
        /// Observation buffer policy.
        buffer: EventBufferConfig,
        /// The single observation receiver handed to the first `events()` caller.
        events: Mutex<Option<mpsc::UnboundedReceiver<HarnessEvent>>>,
        /// Count of events delivered by the cursor.
        delivered: Arc<AtomicUsize>,
        /// Whether the cursor was closed.
        closed: Arc<AtomicBool>,
        /// When set, `send` fails with a protocol error.
        fail_send: AtomicBool,
    }
    impl FakeHarness {
        /// Builds the fake and the sender that scripts its observation stream.
        fn new() -> (Arc<Self>, mpsc::UnboundedSender<HarnessEvent>) {
            let (sender, receiver) = mpsc::unbounded_channel();
            let fake = Arc::new(Self {
                state: Mutex::new(HarnessState::Terminated),
                calls: Arc::new(Mutex::new(Vec::new())),
                card: HarnessCard::new("fake", "0.0.0"),
                buffer: EventBufferConfig::default(),
                events: Mutex::new(Some(receiver)),
                delivered: Arc::new(AtomicUsize::new(0)),
                closed: Arc::new(AtomicBool::new(false)),
                fail_send: AtomicBool::new(false),
            });
            (fake, sender)
        }
        /// Returns the recorded commands.
        fn calls(&self) -> Vec<Call> {
            self.calls.lock().clone()
        }
    }
    /// Channel-backed cursor that records delivery and close.
    struct TestCursor {
        /// Event source, closed when the sender drops.
        receiver: mpsc::UnboundedReceiver<HarnessEvent>,
        /// Shared delivery counter.
        delivered: Arc<AtomicUsize>,
        /// Shared close flag.
        closed: Arc<AtomicBool>,
    }
    #[async_trait]
    impl HarnessEventCursor for TestCursor {
        async fn next_event(&mut self) -> HarnessResult<Option<HarnessEvent>> {
            let event = self.receiver.recv().await;
            if event.is_some() {
                self.delivered.fetch_add(1, Ordering::SeqCst);
            }
            Ok(event)
        }
        async fn close(&mut self) -> HarnessResult<()> {
            self.closed.store(true, Ordering::SeqCst);
            Ok(())
        }
    }
    #[async_trait]
    #[allow(clippy::needless_pass_by_value)]
    impl HarnessProtocol for FakeHarness {
        fn card(&self) -> &HarnessCard {
            &self.card
        }
        fn state(&self) -> HarnessState {
            *self.state.lock()
        }
        fn provider_session_id(&self) -> Option<&str> {
            None
        }
        fn event_buffer_config(&self) -> &EventBufferConfig {
            &self.buffer
        }
        async fn start(&self, context: HarnessContext) -> HarnessResult<()> {
            self.calls.lock().push(Call::Start {
                host_session_id: context.host_session_id,
            });
            *self.state.lock() = HarnessState::Idle;
            Ok(())
        }
        async fn stop(&self) -> HarnessResult<()> {
            self.calls.lock().push(Call::Stop);
            *self.state.lock() = HarnessState::Terminated;
            Ok(())
        }
        fn events(&self) -> Box<dyn HarnessEventCursor> {
            let receiver = self.events.lock().take();
            match receiver {
                Some(receiver) => Box::new(TestCursor {
                    receiver,
                    delivered: Arc::clone(&self.delivered),
                    closed: Arc::clone(&self.closed),
                }),
                None => Box::new(VecEventCursor::new(Vec::new())),
            }
        }
        fn turn_events(&self, _turn_id: Option<&str>) -> Box<dyn HarnessEventCursor> {
            Box::new(VecEventCursor::new(Vec::new()))
        }
        async fn send(
            &self,
            _input: HarnessInput,
            mode: DeliveryMode,
        ) -> HarnessResult<SendReceipt> {
            if self.fail_send.load(Ordering::SeqCst) {
                return Err(HarnessError::state("no cycle is running"));
            }
            self.calls.lock().push(Call::Send { mode });
            Ok(SendReceipt {
                message_id: "message-1".to_owned(),
                turn_id: "turn-1".to_owned(),
                accepted_mode: mode,
            })
        }
        async fn abort(&self, mode: AbortMode) -> HarnessResult<()> {
            self.calls.lock().push(Call::Abort { mode });
            Ok(())
        }
        async fn pause(&self) -> HarnessResult<()> {
            self.calls.lock().push(Call::Pause);
            *self.state.lock() = HarnessState::Paused;
            Ok(())
        }
        async fn resume(&self, query: Option<HarnessInput>) -> HarnessResult<()> {
            self.calls.lock().push(Call::Resume {
                has_query: query.is_some(),
            });
            *self.state.lock() = HarnessState::Running;
            Ok(())
        }
        async fn export_checkpoint(
            &self,
        ) -> HarnessResult<Option<openjiuwen_harness_protocol::HarnessCheckpoint>> {
            Ok(None)
        }
    }
    /// Observer that records callbacks as `"kind:value"` strings.
    #[derive(Default)]
    struct RecordingObserver {
        /// Recorded callbacks, in order.
        calls: Mutex<Vec<String>>,
    }
    impl RecordingObserver {
        /// Returns the recorded callbacks.
        fn calls(&self) -> Vec<String> {
            self.calls.lock().clone()
        }
        /// Appends one callback label.
        fn push(&self, label: String) {
            self.calls.lock().push(label);
        }
    }
    /// Returns the JSON wire string of a serializable value.
    fn wire<T: serde::Serialize>(value: &T) -> String {
        serde_json::to_value(value)
            .ok()
            .and_then(|value| value.as_str().map(str::to_owned))
            .unwrap_or_default()
    }
    impl MemberObserver for RecordingObserver {
        fn on_state(&self, state: HarnessState) {
            self.push(format!("state:{}", wire(&state)));
        }
        fn on_round(&self, round: RoundOutcome) {
            self.push(format!("round:{}", wire(&round)));
        }
        fn on_chunk(&self, chunk: serde_json::Value) {
            let kind = chunk
                .get("type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_owned();
            self.push(format!("chunk:{kind}"));
        }
    }
    /// Builds a runtime over `fake` with one recording observer.
    fn runtime(fake: &Arc<FakeHarness>) -> (HarnessMemberRuntime, Arc<RecordingObserver>) {
        let concrete: Arc<FakeHarness> = Arc::clone(fake);
        let harness: Arc<dyn HarnessProtocol> = concrete;
        let runtime = HarnessMemberRuntime::new(harness, "team:dev-1");
        let observer = Arc::new(RecordingObserver::default());
        runtime.subscribe(Arc::clone(&observer) as Arc<dyn MemberObserver>);
        (runtime, observer)
    }
    /// Polls `condition` for a bounded time, then asserts it held.
    async fn wait_until(mut condition: impl FnMut() -> bool) {
        for _ in 0..500 {
            if condition() {
                return;
            }
            tokio::task::yield_now().await;
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        assert!(condition(), "condition was not met in time");
    }
    /// Builds a `state_changed` observation.
    fn state_event(old: HarnessState, new: HarnessState) -> HarnessEvent {
        HarnessEvent::new(
            1,
            1.0,
            StateChangedEvent { old, new },
            "team:dev-1",
            "dev-1",
        )
    }
    /// Builds a turn-lifecycle observation.
    fn turn_event(kind: TurnEventKind) -> HarnessEvent {
        HarnessEvent::new(
            2,
            2.0,
            TurnLifecycleEvent { kind, result: None },
            "team:dev-1",
            "dev-1",
        )
        .with_turn_id("turn-1")
    }
    /// Builds an answer output observation.
    fn output_event() -> HarnessEvent {
        HarnessEvent::new(
            3,
            3.0,
            OutputEvent {
                output_id: "out-1".to_owned(),
                kind: OutputKind::Text,
                content: json!("hello"),
                operation: OutputOperation::Delta,
                channel: OutputChannel::Answer,
                content_index: 0,
                content_type: None,
                data: std::collections::BTreeMap::new(),
            },
            "team:dev-1",
            "dev-1",
        )
        .with_turn_id("turn-1")
    }
    #[tokio::test]
    async fn state_maps_through_verbatim() {
        let (fake, _sender) = FakeHarness::new();
        *fake.state.lock() = HarnessState::Running;
        let (runtime, _observer) = runtime(&fake);
        assert_eq!(runtime.state(), HarnessState::Running);
        assert_eq!(
            super::super::mapping::state_to_member_status(runtime.state()),
            Some(crate::schema::MemberStatus::Busy)
        );
        assert_eq!(runtime.session_id(), "team:dev-1");
    }
    #[tokio::test]
    async fn start_notifies_ready_and_starts_the_cycle() {
        let (fake, _sender) = FakeHarness::new();
        let (runtime, observer) = runtime(&fake);
        runtime.start(None).await.unwrap();
        assert_eq!(
            fake.calls().first(),
            Some(&Call::Start {
                host_session_id: "team:dev-1".to_owned()
            })
        );
        assert_eq!(runtime.state(), HarnessState::Idle);
        assert_eq!(observer.calls(), vec!["state:idle"]);
        runtime.stop().await.unwrap();
    }
    #[tokio::test]
    async fn send_maps_immediate_to_delivery_mode() {
        let (fake, _sender) = FakeHarness::new();
        let (runtime, _observer) = runtime(&fake);
        runtime.start(None).await.unwrap();
        runtime.send("first".to_owned(), true).await.unwrap();
        runtime.send("second".to_owned(), false).await.unwrap();
        assert_eq!(
            fake.calls(),
            vec![
                Call::Start {
                    host_session_id: "team:dev-1".to_owned()
                },
                Call::Send {
                    mode: DeliveryMode::Steer
                },
                Call::Send {
                    mode: DeliveryMode::FollowUp
                },
            ]
        );
        runtime.stop().await.unwrap();
    }
    #[tokio::test]
    async fn abort_maps_to_force_and_graceful() {
        let (fake, _sender) = FakeHarness::new();
        let (runtime, _observer) = runtime(&fake);
        runtime.start(None).await.unwrap();
        runtime.abort(true).await.unwrap();
        runtime.abort(false).await.unwrap();
        let calls = fake.calls();
        assert_eq!(
            calls[1],
            Call::Abort {
                mode: AbortMode::Force
            }
        );
        assert_eq!(
            calls[2],
            Call::Abort {
                mode: AbortMode::Graceful
            }
        );
        runtime.stop().await.unwrap();
    }
    #[tokio::test]
    async fn pause_and_resume_forward_the_query() {
        let (fake, _sender) = FakeHarness::new();
        let (runtime, _observer) = runtime(&fake);
        runtime.start(None).await.unwrap();
        runtime.pause().await.unwrap();
        assert_eq!(runtime.state(), HarnessState::Paused);
        runtime.resume(Some("continue".to_owned())).await.unwrap();
        runtime.resume(None).await.unwrap();
        let calls = fake.calls();
        assert_eq!(calls[1], Call::Pause);
        assert_eq!(calls[2], Call::Resume { has_query: true });
        assert_eq!(calls[3], Call::Resume { has_query: false });
        runtime.stop().await.unwrap();
    }
    #[tokio::test]
    async fn pump_dispatches_events_in_order_and_closes_the_cursor() {
        let (fake, sender) = FakeHarness::new();
        let (runtime, observer) = runtime(&fake);
        runtime.start(None).await.unwrap();
        sender
            .send(state_event(HarnessState::Idle, HarnessState::Running))
            .unwrap();
        sender.send(turn_event(TurnEventKind::Started)).unwrap();
        sender.send(output_event()).unwrap();
        sender.send(turn_event(TurnEventKind::Finished)).unwrap();
        sender
            .send(state_event(HarnessState::Running, HarnessState::Idle))
            .unwrap();
        drop(sender);
        wait_until(|| observer.calls().len() >= 6).await;
        assert_eq!(
            observer.calls(),
            vec![
                "state:idle",
                "state:running",
                "round:started",
                "chunk:llm_output",
                "round:finished",
                "state:idle",
            ]
        );
        wait_until(|| fake.closed.load(Ordering::SeqCst)).await;
        assert_eq!(fake.delivered.load(Ordering::SeqCst), 5);
        runtime.stop().await.unwrap();
    }
    #[tokio::test]
    async fn stop_cancels_the_pump_and_stops_the_cycle() {
        let (fake, sender) = FakeHarness::new();
        let (runtime, observer) = runtime(&fake);
        runtime.start(None).await.unwrap();
        assert_eq!(observer.calls(), vec!["state:idle"]);
        runtime.stop().await.unwrap();
        let after_stop = observer.calls().len();
        assert_eq!(runtime.state(), HarnessState::Terminated);
        // The pump dropped the cursor, so the scripted channel has no receiver left; either
        // outcome (send error or silently unobserved event) proves the pump stopped.
        let _ = sender.send(state_event(HarnessState::Idle, HarnessState::Running));
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert_eq!(observer.calls().len(), after_stop);
        assert!(fake.calls().contains(&Call::Stop));
    }
    #[tokio::test]
    async fn harness_errors_become_external_errors() {
        let (fake, _sender) = FakeHarness::new();
        let (runtime, _observer) = runtime(&fake);
        runtime.start(None).await.unwrap();
        fake.fail_send.store(true, Ordering::SeqCst);
        let error = runtime.send("x".to_owned(), true).await.unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::External);
        assert!(error.to_string().contains("harness state error"));
        runtime.stop().await.unwrap();
    }
    #[tokio::test]
    async fn empty_session_id_is_rejected() {
        let (fake, _sender) = FakeHarness::new();
        let concrete: Arc<FakeHarness> = Arc::clone(&fake);
        let harness: Arc<dyn HarnessProtocol> = concrete;
        let runtime = HarnessMemberRuntime::new(harness, String::new());
        let error = runtime.start(None).await.unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
    }
}
