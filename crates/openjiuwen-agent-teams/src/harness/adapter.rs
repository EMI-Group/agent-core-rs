//! The adapter from a member's harness protocol to the team's member-runtime seam.
//!
//! Length note: this file is deliberately long because it pairs the adapter with its unit tests
//! (roughly half the lines are `#[cfg(test)]`); the runtime-side pieces it depends on live in
//! `checkpoint`, `checkpoint_store` and `mapping`.
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
//! is owned by a `PumpHandle`: it ends when the cancellation token fires (`stop` or drop of the
//! runtime), when the cursor is exhausted, on a cursor error (logged), or when the harness reports
//! [`HarnessState::Terminated`]. The cursor itself is the cycle-long single-consumer view, so a
//! finished turn does not end the pump.
//!
//! ## Checkpointing
//!
//! A runtime built with a [`MemberCheckpointStore`] always hands its harness a
//! [`MemberCheckpointSink`], and — when its [`MemberCheckpointPolicy`] is enabled — restores the
//! record stored for its `(session_id, card_id)` identity and captures new ones at the configured
//! boundaries (finished turns, state changes, stop). Capture failures are logged and never fail a
//! command or end the pump: a checkpoint is a convenience, not a correctness requirement.
use super::checkpoint::NativeCheckpoint;
use super::checkpoint_store::{
    MemberCheckpointPolicy, MemberCheckpointRecord, MemberCheckpointSink, MemberCheckpointStore,
};
use super::mapping::{chunk_from_event, is_terminal_event, round_from_event, state_from_event};
use crate::member::{MemberObserver, MemberRuntime};
use crate::schema::MemberMode;
use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_harness_protocol::{
    json_object_to_value, AbortMode, DeliveryMode, HarnessContext, HarnessError, HarnessEvent,
    HarnessInput, HarnessProtocol, HarnessState,
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
    /// Identifier of the agent card the member was built from.
    ///
    /// Empty when the team build path supplies none. A changed card invalidates a restored
    /// checkpoint, so this value is part of the member's checkpoint identity.
    pub card_id: String,
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
            card_id: String::new(),
        }
    }
    /// Pins the identifier of the agent card the member was built from.
    #[must_use]
    pub fn with_card_id(mut self, card_id: impl Into<String>) -> Self {
        self.card_id = card_id.into();
        self
    }
    /// Returns the documented per-member session id `"<host_session_id>:<member_name>"`.
    ///
    /// The team session id is shared by every member, so a member's own harness session id is
    /// derived by appending the member name. The runtime wiring passes this derived id as
    /// `MemberHarnessContext::session_id` and the team session as the harness `host_session_id`.
    #[must_use]
    pub fn member_session_id(host_session_id: &str, member_name: &str) -> String {
        format!("{host_session_id}:{member_name}")
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
    /// Identifier of the agent card this member runs; part of its checkpoint identity.
    card_id: String,
    /// Destination of captured checkpoints; `None` disables all persistence.
    checkpoint_store: Option<Arc<dyn MemberCheckpointStore>>,
    /// Which boundaries capture a checkpoint and how a stored one is consumed.
    checkpoint_policy: MemberCheckpointPolicy,
    /// Working directory recorded in every payload and handed to the harness.
    cwd: Option<String>,
    /// Newest captured payload, shared with the pump's capture handle.
    latest_checkpoint: Arc<Mutex<Option<NativeCheckpoint>>>,
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
            card_id: String::new(),
            checkpoint_store: None,
            checkpoint_policy: MemberCheckpointPolicy::default(),
            cwd: None,
            latest_checkpoint: Arc::new(Mutex::new(None)),
        }
    }
    /// Pins the identifier of the agent card this member runs.
    #[must_use]
    pub fn with_card_id(mut self, card_id: impl Into<String>) -> Self {
        self.card_id = card_id.into();
        self
    }
    /// Persists captured checkpoints into `store`.
    #[must_use]
    pub fn with_checkpoint_store(mut self, store: Arc<dyn MemberCheckpointStore>) -> Self {
        self.checkpoint_store = Some(store);
        self
    }
    /// Chooses which boundaries capture a checkpoint and how a stored one is consumed.
    #[must_use]
    pub fn with_checkpoint_policy(mut self, policy: MemberCheckpointPolicy) -> Self {
        self.checkpoint_policy = policy;
        self
    }
    /// Pins the working directory handed to the harness and recorded in every payload.
    #[must_use]
    pub fn with_cwd(mut self, cwd: impl Into<String>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }
    /// Returns the protocol handle, for callers that need provider-specific controls.
    #[must_use]
    pub fn harness(&self) -> &Arc<dyn HarnessProtocol> {
        &self.harness
    }
    /// Returns the agent-card identity this member checkpoints under.
    #[must_use]
    pub fn card_id(&self) -> &str {
        &self.card_id
    }
    /// Returns the checkpoint store, when one is configured.
    #[must_use]
    pub fn checkpoint_store(&self) -> Option<&Arc<dyn MemberCheckpointStore>> {
        self.checkpoint_store.as_ref()
    }
    /// Returns the checkpoint policy (defaults to capture on turn and stop, resume if available).
    #[must_use]
    pub fn checkpoint_policy(&self) -> &MemberCheckpointPolicy {
        &self.checkpoint_policy
    }
    /// Returns the pinned working directory, when one is set.
    #[must_use]
    pub fn cwd(&self) -> Option<&str> {
        self.cwd.as_deref()
    }
    /// Returns the newest checkpoint captured for this member, if any.
    ///
    /// The accessor is `async` on purpose: a provider that cannot expose its checkpoint in memory
    /// would have to read it back from storage, so callers already `.await` this seam.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)] // the async shape is the seam
    pub async fn latest_checkpoint(&self) -> Option<NativeCheckpoint> {
        self.latest_checkpoint.lock().clone()
    }
    /// Captures one checkpoint now, forced, reporting export and save failures.
    ///
    /// Returns `Ok(None)` when no store is configured or the policy is not enabled: in that
    /// configuration no checkpoint work happens at all, which is the default for a runtime built
    /// without a store.
    ///
    /// # Errors
    ///
    /// Returns the provider's error when the export fails, and the store's error when the record
    /// cannot be persisted.
    pub async fn capture_checkpoint(&self) -> Result<Option<NativeCheckpoint>> {
        match self.capture_handle() {
            Some(capture) => capture.capture(None).await,
            None => Ok(None),
        }
    }
    /// Builds the owned capture handle, when checkpointing is configured and enabled.
    fn capture_handle(&self) -> Option<CheckpointCapture> {
        let store = self.checkpoint_store.as_ref()?;
        if !self.checkpoint_policy.is_enabled() {
            return None;
        }
        Some(CheckpointCapture {
            harness: Arc::clone(&self.harness),
            store: Arc::clone(store),
            session_id: self.session_id.clone(),
            card_id: self.card_id.clone(),
            cwd: self.cwd.clone(),
            latest: Arc::clone(&self.latest_checkpoint),
        })
    }
    /// Merges the stored checkpoint into `context` when it is compatible with this member.
    ///
    /// Every failure is logged and leaves `context` cold: restoring is opportunistic, never a
    /// precondition for starting.
    async fn restore_into(&self, mut context: HarnessContext) -> HarnessContext {
        let Some(store) = &self.checkpoint_store else {
            return context;
        };
        match store.load(&self.session_id, &self.card_id).await {
            Ok(Some(record)) => {
                if let Err(error) = record
                    .checkpoint
                    .validate_compatible(&self.session_id, &self.card_id)
                {
                    warn!(
                        error = %error,
                        member = %self.session_id,
                        "stored member checkpoint does not belong to this member; starting cold"
                    );
                    return context;
                }
                if let Some(envelope) = record.protocol {
                    context = context.with_checkpoint(envelope);
                }
                *self.latest_checkpoint.lock() = Some(record.checkpoint);
            }
            Ok(None) => {}
            Err(error) => warn!(
                error = %error,
                member = %self.session_id,
                "stored member checkpoint cannot be read; starting cold"
            ),
        }
        context
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
    ///
    /// `capture` is `Some` only when checkpointing is configured and enabled; the two flags pick
    /// which observation boundaries capture. Captures run after the observers were notified and
    /// never fail the pump.
    fn start_pump(
        &self,
        capture: Option<CheckpointCapture>,
        capture_on_turn: bool,
        capture_on_state: bool,
    ) {
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
                            if let Some(capture) = &capture {
                                if capture_on_turn && is_terminal_event(&event) {
                                    capture.capture_best_effort(event.turn_id.as_deref()).await;
                                } else if capture_on_state && state_from_event(&event).is_some() {
                                    capture.capture_best_effort(None).await;
                                }
                            }
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
    /// Checkpoints: a pinned [`cwd`](Self::with_cwd) is forwarded, and when a
    /// [`checkpoint_store`](Self::with_checkpoint_store) is configured the harness *always* gets a
    /// [`MemberCheckpointSink`] writing back into it plus the policy's resume policy. The stored
    /// record for this member's `(session_id, card_id)` is restored only when the
    /// [`policy`](Self::with_checkpoint_policy) is enabled: a compatible record contributes its
    /// provider envelope (a payload-only record has nothing to resume from, so the harness starts
    /// cold) and seeds [`latest_checkpoint`](Self::latest_checkpoint), while a mismatched,
    /// unreadable or absent record degrades to a cold start — `start` never fails because of
    /// checkpoint loading.
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
        let mut context = HarnessContext::new(
            self.session_id.clone(),
            self.session_id.clone(),
            host_session_id,
            String::new(),
        );
        if let Some(cwd) = &self.cwd {
            context = context.with_cwd(cwd.clone());
        }
        if let Some(store) = &self.checkpoint_store {
            let sink = MemberCheckpointSink::new(
                Arc::clone(store),
                self.session_id.clone(),
                self.card_id.clone(),
            );
            let sink = match &self.cwd {
                Some(cwd) => sink.with_cwd(cwd.clone()),
                None => sink,
            };
            context = context
                .with_checkpoint_sink(Arc::new(sink))
                .with_resume_policy(self.checkpoint_policy.resume_policy());
            if self.checkpoint_policy.is_enabled() {
                context = self.restore_into(context).await;
            }
        }
        let capture = self.capture_handle();
        let capture_on_turn = self.checkpoint_policy.captures_on_turn();
        let capture_on_state = self.checkpoint_policy.captures_on_state();
        self.harness.start(context).await.map_err(harness_error)?;
        self.publish_state();
        self.start_pump(capture, capture_on_turn, capture_on_state);
        Ok(())
    }
    /// Captures a final checkpoint, cancels the pump, then stops the harness cycle.
    ///
    /// The final capture happens before the pump is cancelled and before the harness is stopped, so
    /// it records the pre-stop state with no turn id; a failure is logged and never returned.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::External`] when the provider cannot close
    /// its cycle cleanly.
    async fn stop(&self) -> Result<()> {
        if self.checkpoint_policy.captures_on_stop() {
            if let Some(capture) = self.capture_handle() {
                capture.capture_best_effort(None).await;
            }
        }
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
/// Owned capture state, shared by the pump, [`HarnessMemberRuntime::stop`] and the public capture
/// method.
///
/// The runtime itself is not held behind an `Arc`, so the spawned pump cannot borrow it: this
/// handle carries the owned clones one capture needs (the protocol handle, the store, the member
/// identity and the payload cache).
struct CheckpointCapture {
    /// Protocol handle the provider envelope is exported from.
    harness: Arc<dyn HarnessProtocol>,
    /// Destination of every captured record.
    store: Arc<dyn MemberCheckpointStore>,
    /// Member session identity records are stored under.
    session_id: String,
    /// Agent-card identity records are stored under.
    card_id: String,
    /// Working directory recorded in every payload, when pinned.
    cwd: Option<String>,
    /// Newest payload, shared with [`HarnessMemberRuntime::latest_checkpoint`].
    latest: Arc<Mutex<Option<NativeCheckpoint>>>,
}
impl CheckpointCapture {
    /// Exports the provider envelope and persists one record.
    ///
    /// A `None` export still persists a payload-only record (empty outputs), which is a legitimate
    /// snapshot of a harness that does not expose its own checkpoints.
    async fn capture(&self, turn_id: Option<&str>) -> Result<Option<NativeCheckpoint>> {
        let envelope = self
            .harness
            .export_checkpoint()
            .await
            .map_err(harness_error)?;
        let outputs = envelope.as_ref().map_or_else(
            || serde_json::json!({}),
            |checkpoint| json_object_to_value(&checkpoint.data),
        );
        let mut payload = NativeCheckpoint::new(self.session_id.clone(), self.card_id.clone())
            .with_state(self.harness.state())
            .with_outputs(outputs);
        if let Some(cwd) = &self.cwd {
            payload = payload.with_cwd(cwd.clone());
        }
        if let Some(turn_id) = turn_id {
            payload = payload.with_turn_id(turn_id.to_owned());
        }
        let mut record = MemberCheckpointRecord::new(payload.clone());
        if let Some(envelope) = envelope {
            record = record.with_protocol(envelope);
        }
        let revision = self.store.save(record).await?;
        debug!(
            revision = %revision,
            member = %self.session_id,
            card = %self.card_id,
            "member checkpoint captured"
        );
        *self.latest.lock() = Some(payload.clone());
        Ok(Some(payload))
    }
    /// [`capture`](Self::capture) with every failure logged and ignored.
    ///
    /// This is what the pump and `stop` use: an export or storage failure must never fail a member
    /// command or end the observation pump.
    async fn capture_best_effort(&self, turn_id: Option<&str>) {
        if let Err(error) = self.capture(turn_id).await {
            warn!(
                error = %error,
                member = %self.session_id,
                "member checkpoint capture failed"
            );
        }
    }
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
    use crate::harness::checkpoint::NativeCheckpoint;
    use crate::harness::checkpoint_store::{
        InMemoryMemberCheckpointStore, MemberCheckpointPolicy, MemberCheckpointRecord,
    };
    use crate::schema::RoundOutcome;
    use openjiuwen_harness_protocol::{
        EventBufferConfig, HarnessCard, HarnessCheckpoint, HarnessError, HarnessEventCursor,
        HarnessResult, HarnessState, JsonObject, OutputChannel, OutputEvent, OutputKind,
        OutputOperation, ResumePolicy, SendReceipt, StateChangedEvent, TurnEventKind,
        TurnLifecycleEvent, VecEventCursor,
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
        /// Contexts received by `start`, in order (`HarnessContext` is not `Debug`, so this is a
        /// list rather than an assertable value).
        contexts: Mutex<Vec<HarnessContext>>,
        /// Envelope returned by `export_checkpoint`.
        checkpoint: Mutex<Option<HarnessCheckpoint>>,
        /// When set, `export_checkpoint` fails.
        export_fails: AtomicBool,
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
                contexts: Mutex::new(Vec::new()),
                checkpoint: Mutex::new(None),
                export_fails: AtomicBool::new(false),
            });
            (fake, sender)
        }
        /// Returns the recorded commands.
        fn calls(&self) -> Vec<Call> {
            self.calls.lock().clone()
        }
        /// Returns the contexts `start` received, in order.
        fn contexts(&self) -> Vec<HarnessContext> {
            self.contexts.lock().clone()
        }
        /// Returns the context the most recent `start` received.
        fn last_context(&self) -> Option<HarnessContext> {
            self.contexts.lock().last().cloned()
        }
        /// Scripts the envelope `export_checkpoint` returns.
        fn set_checkpoint(&self, checkpoint: Option<HarnessCheckpoint>) {
            *self.checkpoint.lock() = checkpoint;
        }
        /// Makes `export_checkpoint` fail.
        fn set_export_fails(&self, fails: bool) {
            self.export_fails.store(fails, Ordering::SeqCst);
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
                host_session_id: context.host_session_id.clone(),
            });
            self.contexts.lock().push(context);
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
        async fn export_checkpoint(&self) -> HarnessResult<Option<HarnessCheckpoint>> {
            if self.export_fails.load(Ordering::SeqCst) {
                return Err(HarnessError::protocol("checkpoint export unavailable"));
            }
            Ok(self.checkpoint.lock().clone())
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
    /// Session every checkpoint-enabled runtime in these tests runs under.
    const MEMBER_SESSION: &str = "team:dev-1";
    /// Agent-card identity every checkpoint-enabled runtime in these tests uses.
    const MEMBER_CARD: &str = "card-1";
    /// Builds an empty in-memory store, kept concrete so tests can inspect it.
    fn memory_store() -> Arc<InMemoryMemberCheckpointStore> {
        Arc::new(InMemoryMemberCheckpointStore::new())
    }
    /// Coerces the concrete store into the trait handle a runtime takes.
    fn shared(store: &Arc<InMemoryMemberCheckpointStore>) -> Arc<dyn MemberCheckpointStore> {
        let concrete: Arc<InMemoryMemberCheckpointStore> = Arc::clone(store);
        concrete
    }
    /// Builds a runtime over `fake` with a store, a card id and `policy`.
    fn checkpoint_runtime(
        fake: &Arc<FakeHarness>,
        store: Arc<dyn MemberCheckpointStore>,
        policy: MemberCheckpointPolicy,
    ) -> (HarnessMemberRuntime, Arc<RecordingObserver>) {
        let concrete: Arc<FakeHarness> = Arc::clone(fake);
        let harness: Arc<dyn HarnessProtocol> = concrete;
        let runtime = HarnessMemberRuntime::new(harness, MEMBER_SESSION)
            .with_card_id(MEMBER_CARD)
            .with_checkpoint_store(store)
            .with_checkpoint_policy(policy);
        let observer = Arc::new(RecordingObserver::default());
        runtime.subscribe(Arc::clone(&observer) as Arc<dyn MemberObserver>);
        (runtime, observer)
    }
    /// Builds the provider envelope the fake exports, carrying one answer part.
    fn envelope(session_id: &str) -> HarnessCheckpoint {
        let data: JsonObject = [("answer_parts".to_owned(), json!(["hi"]))]
            .into_iter()
            .collect();
        HarnessCheckpoint {
            provider: "fake".to_owned(),
            schema_version: "1".to_owned(),
            agent_id: session_id.to_owned(),
            host_session_id: session_id.to_owned(),
            checkpoint_id: "cp-1".to_owned(),
            sequence: 1,
            data,
            provider_session_id: None,
            revision: None,
        }
    }
    #[tokio::test]
    async fn start_attaches_a_sink_only_when_a_store_is_configured() {
        let (plain_fake, _plain_sender) = FakeHarness::new();
        let (plain, _plain_observer) = runtime(&plain_fake);
        plain.start(None).await.unwrap();
        let plain_context = plain_fake.last_context().expect("start context");
        assert!(plain_context.checkpoint_sink.is_none());
        assert_eq!(plain_context.resume_policy, ResumePolicy::New);
        plain.stop().await.unwrap();

        let store = memory_store();
        let (stored_fake, _stored_sender) = FakeHarness::new();
        let (stored, _stored_observer) = checkpoint_runtime(
            &stored_fake,
            shared(&store),
            MemberCheckpointPolicy::default(),
        );
        stored.start(None).await.unwrap();
        let stored_context = stored_fake.last_context().expect("start context");
        assert!(stored_context.checkpoint_sink.is_some());
        assert_eq!(
            stored_context.resume_policy,
            ResumePolicy::ResumeIfAvailable
        );
        stored.stop().await.unwrap();
    }
    #[tokio::test]
    async fn start_restores_the_stored_envelope_and_policy() {
        let store = memory_store();
        let envelope = envelope(MEMBER_SESSION);
        let payload = NativeCheckpoint::new(MEMBER_SESSION, MEMBER_CARD);
        store
            .save(MemberCheckpointRecord::new(payload.clone()).with_protocol(envelope.clone()))
            .await
            .unwrap();

        let (fake, _sender) = FakeHarness::new();
        let (runtime, _observer) =
            checkpoint_runtime(&fake, shared(&store), MemberCheckpointPolicy::default());
        runtime.start(None).await.unwrap();

        assert_eq!(fake.contexts().len(), 1);
        let context = fake.last_context().expect("start context");
        assert_eq!(context.checkpoint, Some(envelope.clone()));
        assert_eq!(context.resume_policy, ResumePolicy::ResumeIfAvailable);
        assert_eq!(context.host_session_id, MEMBER_SESSION);
        assert_eq!(runtime.latest_checkpoint().await, Some(payload));
        runtime.stop().await.unwrap();
    }
    #[tokio::test]
    async fn start_starts_cold_on_a_session_or_card_mismatch() {
        let store = memory_store();
        // Stored under the right key, but the payload belongs to another session.
        store
            .save(
                MemberCheckpointRecord::new(NativeCheckpoint::new("other:dev-1", MEMBER_CARD))
                    .with_protocol(envelope("other:dev-1")),
            )
            .await
            .unwrap();
        let (fake, _sender) = FakeHarness::new();
        let (runtime, _observer) =
            checkpoint_runtime(&fake, shared(&store), MemberCheckpointPolicy::default());
        runtime.start(None).await.unwrap();
        let context = fake.last_context().expect("start context");
        assert!(context.checkpoint.is_none());
        assert_eq!(runtime.state(), HarnessState::Idle);
        assert_eq!(runtime.latest_checkpoint().await, None);
        runtime.stop().await.unwrap();

        // Same intent for a card change: the record is found but not compatible.
        store
            .save(
                MemberCheckpointRecord::new(NativeCheckpoint::new(MEMBER_SESSION, "card-2"))
                    .with_protocol(envelope(MEMBER_SESSION)),
            )
            .await
            .unwrap();
        let (card_fake, _card_sender) = FakeHarness::new();
        let (card_runtime, _card_observer) = checkpoint_runtime(
            &card_fake,
            shared(&store),
            MemberCheckpointPolicy::default(),
        );
        card_runtime.start(None).await.unwrap();
        let card_context = card_fake.last_context().expect("start context");
        assert!(card_context.checkpoint.is_none());
        assert_eq!(card_runtime.state(), HarnessState::Idle);
        card_runtime.stop().await.unwrap();
    }
    #[tokio::test]
    async fn a_turn_boundary_writes_a_checkpoint_that_a_second_runtime_restores() {
        let store = memory_store();
        let envelope = envelope(MEMBER_SESSION);
        let (fake, sender) = FakeHarness::new();
        fake.set_checkpoint(Some(envelope.clone()));
        let (runtime, _observer) =
            checkpoint_runtime(&fake, shared(&store), MemberCheckpointPolicy::default());
        runtime.start(None).await.unwrap();
        sender.send(turn_event(TurnEventKind::Finished)).unwrap();
        wait_until(|| !store.is_empty()).await;

        let record = store.get(MEMBER_SESSION, MEMBER_CARD).expect("record");
        assert_eq!(record.checkpoint.session_id, MEMBER_SESSION);
        assert_eq!(record.checkpoint.card_id, MEMBER_CARD);
        assert_eq!(record.checkpoint.state, HarnessState::Idle);
        assert_eq!(record.checkpoint.turn_id.as_deref(), Some("turn-1"));
        assert_eq!(record.checkpoint.outputs, json!({"answer_parts": ["hi"]}));
        assert_eq!(record.protocol, Some(envelope.clone()));
        runtime.stop().await.unwrap();

        // A fresh harness over the same store restores the persisted envelope.
        let (second_fake, _second_sender) = FakeHarness::new();
        let (second, _second_observer) = checkpoint_runtime(
            &second_fake,
            shared(&store),
            MemberCheckpointPolicy::default(),
        );
        second.start(None).await.unwrap();
        let context = second_fake.last_context().expect("start context");
        assert_eq!(context.checkpoint, Some(envelope));
        second.stop().await.unwrap();
    }
    #[tokio::test]
    async fn stop_writes_a_final_checkpoint() {
        let store = memory_store();
        let (fake, _sender) = FakeHarness::new();
        fake.set_checkpoint(Some(envelope(MEMBER_SESSION)));
        let (runtime, _observer) =
            checkpoint_runtime(&fake, shared(&store), MemberCheckpointPolicy::default());
        runtime.start(None).await.unwrap();
        runtime.stop().await.unwrap();

        assert_eq!(store.len(), 1);
        let record = store.get(MEMBER_SESSION, MEMBER_CARD).expect("record");
        assert_eq!(record.checkpoint.state, HarnessState::Idle);
        assert_eq!(record.checkpoint.turn_id, None);
        assert_eq!(record.protocol, Some(envelope(MEMBER_SESSION)));
    }
    #[tokio::test]
    async fn a_disabled_policy_writes_nothing() {
        let store = memory_store();
        let (fake, sender) = FakeHarness::new();
        fake.set_checkpoint(Some(envelope(MEMBER_SESSION)));
        let (runtime, observer) =
            checkpoint_runtime(&fake, shared(&store), MemberCheckpointPolicy::disabled());
        runtime.start(None).await.unwrap();
        // The store is still wired into the harness context; a disabled policy only stops this
        // runtime from capturing or restoring, so the sink never receives anything.
        assert!(fake
            .last_context()
            .expect("start context")
            .checkpoint_sink
            .is_some());
        sender.send(turn_event(TurnEventKind::Finished)).unwrap();
        wait_until(|| observer.calls().iter().any(|call| call == "round:finished")).await;
        runtime.stop().await.unwrap();

        assert_eq!(store.len(), 0);
        assert_eq!(runtime.latest_checkpoint().await, None);
        assert!(runtime.capture_checkpoint().await.unwrap().is_none());
    }
    #[tokio::test]
    async fn an_export_error_does_not_break_the_pump() {
        let store = memory_store();
        let (fake, sender) = FakeHarness::new();
        fake.set_checkpoint(Some(envelope(MEMBER_SESSION)));
        fake.set_export_fails(true);
        let (runtime, observer) =
            checkpoint_runtime(&fake, shared(&store), MemberCheckpointPolicy::default());
        runtime.start(None).await.unwrap();
        sender.send(turn_event(TurnEventKind::Finished)).unwrap();
        sender
            .send(state_event(HarnessState::Idle, HarnessState::Running))
            .unwrap();
        wait_until(|| observer.calls().iter().any(|call| call == "state:running")).await;

        assert_eq!(store.len(), 0);
        assert_eq!(runtime.latest_checkpoint().await, None);
        runtime.stop().await.unwrap();
    }
    #[tokio::test]
    async fn member_session_id_and_card_id() {
        assert_eq!(
            MemberHarnessContext::member_session_id("team:1", "dev-1"),
            "team:1:dev-1"
        );
        let context = MemberHarnessContext::new(
            "team",
            "dev-1",
            "team:1:dev-1",
            "/work",
            "en",
            MemberMode::BuildMode,
        );
        assert_eq!(context.card_id, "");
        let pinned = context.clone().with_card_id("card-1");
        assert_eq!(pinned.card_id, "card-1");
        assert_eq!(pinned.team_name, context.team_name);
        assert_eq!(pinned.member_name, context.member_name);
        assert_eq!(pinned.session_id, context.session_id);
        assert_eq!(pinned.workspace_root, context.workspace_root);
        assert_eq!(pinned.language, context.language);
        assert_eq!(pinned.mode, context.mode);

        let (fake, _sender) = FakeHarness::new();
        let concrete: Arc<FakeHarness> = Arc::clone(&fake);
        let harness: Arc<dyn HarnessProtocol> = concrete;
        let plain = HarnessMemberRuntime::new(harness, MEMBER_SESSION);
        assert_eq!(plain.card_id(), "");
        assert!(plain.checkpoint_store().is_none());
        assert_eq!(
            plain.checkpoint_policy(),
            &MemberCheckpointPolicy::default()
        );
        assert_eq!(plain.cwd(), None);
        let pinned_runtime = plain.with_card_id("card-1").with_cwd("/work/dev-1");
        assert_eq!(pinned_runtime.card_id(), "card-1");
        assert_eq!(pinned_runtime.cwd(), Some("/work/dev-1"));
        assert!(pinned_runtime.checkpoint_store().is_none());
        assert!(pinned_runtime.latest_checkpoint().await.is_none());
        assert!(pinned_runtime.capture_checkpoint().await.unwrap().is_none());
    }
}
