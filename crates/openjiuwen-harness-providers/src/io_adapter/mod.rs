//! The harness IO adapter: projects a harness onto the kernel agent IO surface.
//!
//! [`HarnessIOAdapter`] wraps a [`DynHarness`] and turns it into the shape the openJiuwen kernel
//! expects from an agent: a lifecycle (`start`/`stop`/`pause`/`resume`/`abort`), a single
//! streaming consumer ([`HarnessIOAdapter::outputs`]) of [`AgentChunk`] frames, input delivery
//! ([`HarnessIOAdapter::send`]) with the kernel's interactive-input contract, and a
//! [`HarnessInteractionHandler`] implementation that answers the harness' awaited interactions
//! (tool approvals, user input, MCP elicitations, dynamic tool calls, provider requests).
//!
//! Python parity: `openjiuwen.harness_providers.io_adapter`. The Python adapter exposes one async
//! generator fed by a background pump task; Rust exposes the same single-consumer contract as a
//! boxed [`futures::Stream`].
//!
//! # Structure
//!
//! - this module — the adapter, its configuration, the two host-service traits, lifecycle,
//!   delivery and the interaction-handler delegation;
//! - [`projection`] — [`HarnessEvent`] to [`AgentChunk`] projection;
//! - [`interactions`] — interaction routing, the pending-interaction map and its resolution;
//! - [`answers`] — parsing host answers into approval and user-input responses.
//!
//! # Concurrency
//!
//! Every shared field is either atomic or behind a [`std::sync::Mutex`]; no guard is ever held
//! across an `.await`. Exactly one event-pump task is spawned per successful
//! [`HarnessIOAdapter::start`] and it is owned (and aborted) by the adapter, so no detached task
//! outlives the adapter. The pump holds the adapter weakly: an adapter dropped without
//! [`HarnessIOAdapter::stop`] ends the pump (releasing the cursor) instead of keeping itself
//! alive through the task's handle.

mod answers;
mod interactions;
mod projection;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use async_trait::async_trait;
use futures::stream::BoxStream;
use openjiuwen_agent::AgentChunk;
use openjiuwen_harness_protocol::{
    AbortMode, DeliveryMode, DynHarness, HarnessCapability, HarnessContext, HarnessError,
    HarnessEvent, HarnessInput, HarnessInteractionHandler, HarnessInteractionRequest,
    HarnessInteractionResponse, HarnessResult, HarnessState, HostCapability,
    InteractionCancelReason, JsonValue, ProviderInteractionRequest, ProviderInteractionResponse,
    SendReceipt, ToolGateway,
};
use openjiuwen_session::interaction::InteractiveInput;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

use crate::input::{to_harness_input, INTERACTIVE_INPUT_KIND};
use projection::Projector;

/// Observer called with every harness event the adapter consumes.
///
/// The observer runs before projection and is infallible: it is an observation hook (Python's
/// `event_observer` callback), not a control surface. Implementations must not block the async
/// executor.
#[async_trait]
pub trait EventObserver: Send + Sync {
    /// Observes one harness event.
    async fn observe(&self, event: &HarnessEvent);
}

/// Handler answering provider-specific interaction requests.
///
/// The adapter answers shared interactions itself (approvals, user input, elicitations, dynamic
/// tool calls) and delegates only [`ProviderInteractionRequest`]s, which are provider-defined.
#[async_trait]
pub trait ProviderInteractionHandler: Send + Sync {
    /// Returns the response for `request`.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError`] when the handler cannot answer the request.
    async fn handle(
        &self,
        request: &ProviderInteractionRequest,
    ) -> HarnessResult<ProviderInteractionResponse>;
}

/// Configuration of a [`HarnessIOAdapter`].
///
/// The defaults match Python's adapter: no observer, no turn-lifecycle or turn-failure chunks,
/// automatic tool approval, and no stopping on an unsupported force abort.
#[derive(Clone)]
#[allow(clippy::struct_excessive_bools)] // mirrors Python's boolean config switches
pub struct HarnessIOAdapterConfig {
    /// Observer called with every harness event before it is projected.
    pub event_observer: Option<Arc<dyn EventObserver>>,
    /// Whether turn lifecycle events become `turn_lifecycle` chunks. Defaults to `false`.
    pub emit_turn_lifecycle: bool,
    /// Whether a failed turn becomes a `task_failed` chunk when turn-lifecycle chunks are off.
    pub emit_turn_failure: bool,
    /// Whether tool approvals are answered with `Allow` without asking the host. Defaults to
    /// `true`.
    pub auto_approve_tools: bool,
    /// Whether an unsupported force abort stops the harness instead of failing. Defaults to
    /// `false`.
    pub stop_on_unsupported_force_abort: bool,
    /// Handler answering provider-specific interaction requests, when one is configured.
    pub provider_interaction_handler: Option<Arc<dyn ProviderInteractionHandler>>,
}

impl Default for HarnessIOAdapterConfig {
    fn default() -> Self {
        Self {
            event_observer: None,
            emit_turn_lifecycle: false,
            emit_turn_failure: false,
            auto_approve_tools: true,
            stop_on_unsupported_force_abort: false,
            provider_interaction_handler: None,
        }
    }
}

/// Kind of interaction waiting for a host answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PendingKind {
    /// A tool approval, answered with [`ToolApprovalResponse`](openjiuwen_harness_protocol::ToolApprovalResponse).
    ToolApproval,
    /// A user-input request, answered with [`UserInputResponse`](openjiuwen_harness_protocol::UserInputResponse).
    UserInput,
}

/// One interaction waiting for a host answer.
struct PendingInteraction {
    /// Which answer parser the host answer must go through.
    kind: PendingKind,
    /// Delivers the parsed response to the waiting [`HarnessInteractionHandler::handle`] call.
    sender: oneshot::Sender<HarnessInteractionResponse>,
}

/// Projects one harness onto the kernel agent IO surface.
///
/// Build it with [`HarnessIOAdapter::new`] (which returns an [`Arc`] because the adapter is both
/// the stream producer and its own interaction handler) and drive it with [`start`](Self::start).
///
/// [`HarnessIOAdapter::outputs`] is single-consumer: the pump task feeds one channel and the first
/// call takes the receiver, so a second call fails with [`HarnessError::State`].
pub struct HarnessIOAdapter {
    /// The wrapped harness.
    harness: DynHarness,
    /// Adapter configuration.
    config: HarnessIOAdapterConfig,
    /// Whether [`start`](Self::start) succeeded; guards double starts.
    started: AtomicBool,
    /// Whether [`stop`](Self::stop) already ran; makes it idempotent.
    stopped: AtomicBool,
    /// Whether [`outputs`](Self::outputs) already took the receiver.
    outputs_taken: AtomicBool,
    /// Chunk producer side, cloned into the event pump.
    event_tx: mpsc::UnboundedSender<AgentChunk>,
    /// Chunk consumer side, handed out once by [`outputs`](Self::outputs).
    event_rx: Mutex<Option<mpsc::UnboundedReceiver<AgentChunk>>>,
    /// The event-pump task of the current cycle.
    pump: Mutex<Option<JoinHandle<()>>>,
    /// Monotonic chunk index for the adapter lifetime.
    next_index: AtomicI64,
    /// Interactions waiting for a host answer, keyed by request id.
    pending: Mutex<HashMap<String, PendingInteraction>>,
    /// Host tool gateway captured from the last [`prepare_context`](Self::prepare_context) call.
    tool_gateway: Mutex<Option<Arc<dyn ToolGateway>>>,
}

impl HarnessIOAdapter {
    /// Wraps `harness` with `config` and returns the shared adapter.
    ///
    /// The adapter is unstarted: no task is spawned and no harness call is made until
    /// [`start`](Self::start).
    #[must_use]
    pub fn new(harness: DynHarness, config: HarnessIOAdapterConfig) -> Arc<Self> {
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        Arc::new(Self {
            harness,
            config,
            started: AtomicBool::new(false),
            stopped: AtomicBool::new(false),
            outputs_taken: AtomicBool::new(false),
            event_tx,
            event_rx: Mutex::new(Some(event_rx)),
            pump: Mutex::new(None),
            next_index: AtomicI64::new(0),
            pending: Mutex::new(HashMap::new()),
            tool_gateway: Mutex::new(None),
        })
    }

    /// Returns the wrapped harness.
    #[must_use]
    pub fn harness(&self) -> &DynHarness {
        &self.harness
    }

    /// Returns the adapter configuration.
    #[must_use]
    pub fn config(&self) -> &HarnessIOAdapterConfig {
        &self.config
    }

    /// Returns the harness lifecycle state.
    #[must_use]
    pub fn state(&self) -> HarnessState {
        self.harness.state()
    }

    /// Returns the provider-native session id, once the provider has one.
    #[must_use]
    pub fn session_id(&self) -> Option<String> {
        self.harness.provider_session_id().map(str::to_owned)
    }

    /// Returns the sorted ids of the interactions still waiting for a host answer.
    #[must_use]
    pub fn pending_interrupt_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = lock(&self.pending).keys().cloned().collect();
        ids.sort();
        ids
    }

    /// Returns whether any interaction is waiting for a host answer.
    #[must_use]
    pub fn has_pending_interrupt(&self) -> bool {
        !lock(&self.pending).is_empty()
    }

    /// Returns whether `input` can resume the interactions that are currently pending.
    ///
    /// A raw payload resumes only a single pending interaction; otherwise the input must answer at
    /// least one pending request id through `interrupt_answers` or `user_inputs`.
    #[must_use]
    pub fn is_pending_interrupt_resume_valid(&self, input: &InteractiveInput) -> bool {
        let pending = lock(&self.pending);
        if pending.is_empty() {
            return false;
        }
        if input.raw_inputs.is_some() {
            return pending.len() == 1;
        }
        pending.keys().any(|request_id| {
            input.interrupt_answers.contains_key(request_id)
                || input.user_inputs.contains_key(request_id)
        })
    }

    /// Declares the host services this adapter provides and returns the prepared context.
    ///
    /// Always declares [`HostCapability::UserInput`]; adds [`HostCapability::ToolApproval`] unless
    /// approvals are automatic, [`HostCapability::ProviderInteraction`] when a provider handler is
    /// configured, and [`HostCapability::NativeToolGateway`] plus
    /// [`HostCapability::DynamicToolCall`] when the host supplies tools. The host tool gateway is
    /// captured for dynamic tool calls, and the adapter installs itself as the interaction handler
    /// when the context does not name one.
    #[must_use]
    pub fn prepare_context(self: &Arc<Self>, mut context: HarnessContext) -> HarnessContext {
        context.host_capabilities.insert(HostCapability::UserInput);
        if !self.config.auto_approve_tools {
            context
                .host_capabilities
                .insert(HostCapability::ToolApproval);
        }
        if self.config.provider_interaction_handler.is_some() {
            context
                .host_capabilities
                .insert(HostCapability::ProviderInteraction);
        }
        lock(&self.tool_gateway).clone_from(&context.tools);
        if context.tools.is_some() {
            context
                .host_capabilities
                .insert(HostCapability::NativeToolGateway);
            context
                .host_capabilities
                .insert(HostCapability::DynamicToolCall);
        }
        if context.interactions.is_none() {
            let handler: Arc<dyn HarnessInteractionHandler> = self.clone();
            context.interactions = Some(handler);
        }
        context
    }

    /// Starts the harness with the prepared context and spawns the event pump.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::State`] when the adapter is already started, and the harness' own
    /// error when it refuses to start; in that case the adapter is left unstarted and the harness
    /// is stopped best-effort.
    pub async fn start(self: &Arc<Self>, context: HarnessContext) -> HarnessResult<()> {
        if self.started.swap(true, Ordering::SeqCst) {
            return Err(HarnessError::state("harness IO adapter is already started"));
        }
        let context = self.prepare_context(context);
        if let Err(error) = self.harness.start(context).await {
            let _ = self.harness.stop().await;
            self.started.store(false, Ordering::SeqCst);
            return Err(error);
        }

        let mut cursor = self.harness.events();
        let projector = Projector::new(&self.config, self.harness.card().name.clone());
        let observer = self.config.event_observer.clone();
        let adapter = Arc::downgrade(self);
        let event_tx = self.event_tx.clone();
        let handle = tokio::spawn(async move {
            loop {
                match cursor.next_event().await {
                    Ok(Some(event)) => {
                        if let Some(observer) = observer.as_ref() {
                            observer.observe(&event).await;
                        }
                        let Some(adapter) = adapter.upgrade() else {
                            break;
                        };
                        let chunk = projector.project(&event, || adapter.next_chunk_index());
                        if let Some(chunk) = chunk {
                            let _ = event_tx.send(chunk);
                        }
                    }
                    Ok(None) => break,
                    Err(error) => {
                        tracing::warn!(%error, "harness IO adapter event pump failed");
                        break;
                    }
                }
            }
            let _ = cursor.close().await;
        });
        *lock(&self.pump) = Some(handle);
        Ok(())
    }

    /// Stops the harness and tears down the current cycle.
    ///
    /// Idempotent: the second call is a no-op, and stopping is terminal for the adapter instance
    /// (it owns one cycle). The event pump is aborted, every pending interaction is cancelled
    /// (its waiter maps the dropped sender to [`HarnessError::State`]), and only then is the
    /// harness stopped.
    ///
    /// # Errors
    ///
    /// Propagates the harness' own error from stopping.
    pub async fn stop(&self) -> HarnessResult<()> {
        if self.stopped.swap(true, Ordering::SeqCst) {
            return Ok(());
        }
        if let Some(handle) = lock(&self.pump).take() {
            handle.abort();
        }
        lock(&self.pending).clear();
        self.started.store(false, Ordering::SeqCst);
        self.harness.stop().await
    }

    /// Takes the single stream of projected chunks.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::State`] when the stream was already taken; the adapter produces
    /// chunks for exactly one consumer.
    pub fn outputs(self: &Arc<Self>) -> HarnessResult<BoxStream<'static, AgentChunk>> {
        if self.outputs_taken.swap(true, Ordering::SeqCst) {
            return Err(HarnessError::state(
                "harness IO adapter outputs are single-consumer",
            ));
        }
        let receiver = lock(&self.event_rx)
            .take()
            .ok_or_else(|| HarnessError::state("harness IO adapter outputs are single-consumer"))?;
        Ok(Box::pin(futures::stream::unfold(
            receiver,
            |mut receiver| async move { receiver.recv().await.map(|chunk| (chunk, receiver)) },
        )))
    }

    /// Returns the delivery mode for an input while the harness is running.
    ///
    /// A running harness steers when the input must be immediate and the card declares
    /// [`HarnessCapability::Steer`], and queues it as a follow-up otherwise; any other state lets
    /// the harness choose with [`DeliveryMode::Auto`].
    #[must_use]
    pub fn delivery_mode(&self, immediate: bool) -> DeliveryMode {
        if self.state() == HarnessState::Running {
            if immediate && self.harness.card().supports(HarnessCapability::Steer) {
                DeliveryMode::Steer
            } else {
                DeliveryMode::FollowUp
            }
        } else {
            DeliveryMode::Auto
        }
    }

    /// Delivers `content` to the harness.
    ///
    /// A payload carrying host answers (`user_inputs`, `interrupt_answers` or `raw_inputs`)
    /// resolves the matching pending interactions and returns `Ok(None)`; when it resolves
    /// nothing it is forwarded to the harness tagged as [`INTERACTIVE_INPUT_KIND`]. Any other
    /// payload is forwarded verbatim.
    ///
    /// # Errors
    ///
    /// Propagates the harness' delivery error (no running cycle, unsupported delivery mode, …).
    pub async fn send(
        &self,
        content: JsonValue,
        immediate: bool,
    ) -> HarnessResult<Option<SendReceipt>> {
        if let Some(input) = answers::interactive_input(&content) {
            if interactions::resolve_pending(self, &input) > 0 {
                return Ok(None);
            }
            let mut payload = to_harness_input(content);
            payload.metadata.insert(
                "kind".to_owned(),
                JsonValue::String(INTERACTIVE_INPUT_KIND.to_owned()),
            );
            let receipt = self
                .harness
                .send(payload, self.delivery_mode(immediate))
                .await?;
            return Ok(Some(receipt));
        }
        let receipt = self
            .harness
            .send(to_harness_input(content), self.delivery_mode(immediate))
            .await?;
        Ok(Some(receipt))
    }

    /// Aborts the active turn according to the declared capabilities.
    ///
    /// `immediate` is kept for Python signature parity and is not used: both paths follow the
    /// graceful-then-force capability order, and only when neither capability is declared does
    /// this fail (or stop the harness, when
    /// [`stop_on_unsupported_force_abort`](HarnessIOAdapterConfig::stop_on_unsupported_force_abort)
    /// is set).
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::UnsupportedCapability`] when the card declares neither
    /// [`HarnessCapability::GracefulAbort`] nor [`HarnessCapability::ForceAbort`], and propagates
    /// the harness' own abort error.
    pub async fn abort(&self, immediate: bool) -> HarnessResult<()> {
        let _ = immediate;
        let card = self.harness.card();
        let name = card.name.clone();
        let graceful = card.supports(HarnessCapability::GracefulAbort);
        let force = card.supports(HarnessCapability::ForceAbort);
        if graceful {
            self.harness.abort(AbortMode::Graceful).await
        } else if force {
            self.harness.abort(AbortMode::Force).await
        } else if self.config.stop_on_unsupported_force_abort {
            self.stop().await
        } else {
            Err(HarnessError::unsupported_capability(format!(
                "harness '{name}' does not support force_abort"
            )))
        }
    }

    /// Pauses the active turn.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::UnsupportedCapability`] when the card does not declare
    /// [`HarnessCapability::PauseResume`], and propagates the harness' own pause error.
    pub async fn pause(&self) -> HarnessResult<()> {
        let name = self.harness.card().name.clone();
        if !self.harness.card().supports(HarnessCapability::PauseResume) {
            return Err(HarnessError::unsupported_capability(format!(
                "harness '{name}' does not support pause/resume"
            )));
        }
        self.harness.pause().await
    }

    /// Resumes a paused turn, optionally with the input that continues it.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::UnsupportedCapability`] when the card does not declare
    /// [`HarnessCapability::PauseResume`], and propagates the harness' own resume error.
    pub async fn resume(&self, query: Option<HarnessInput>) -> HarnessResult<()> {
        let name = self.harness.card().name.clone();
        if !self.harness.card().supports(HarnessCapability::PauseResume) {
            return Err(HarnessError::unsupported_capability(format!(
                "harness '{name}' does not support pause/resume"
            )));
        }
        self.harness.resume(query).await
    }

    /// Returns the next chunk index for this adapter.
    fn next_chunk_index(&self) -> i64 {
        self.next_index.fetch_add(1, Ordering::SeqCst)
    }
}

#[async_trait]
impl HarnessInteractionHandler for HarnessIOAdapter {
    async fn handle(
        &self,
        request: &HarnessInteractionRequest,
    ) -> HarnessResult<HarnessInteractionResponse> {
        interactions::route(self, request).await
    }

    async fn cancel(&self, request_id: &str, reason: InteractionCancelReason) -> HarnessResult<()> {
        interactions::cancel(self, request_id, reason);
        Ok(())
    }
}

/// Locks `mutex`, recovering from a poisoned lock instead of panicking.
///
/// Every guard is dropped before the next `.await`; recovery keeps a panic in one task from
/// turning adapter state into a permanent error for every later call.
pub(super) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}
