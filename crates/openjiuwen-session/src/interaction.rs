//! Human-in-the-loop interaction requests, responses, and the interaction broker.
//!
//! Maps to `openjiuwen/core/session/interaction/` in the Python reference
//! (v0.1.20). Python interrupts execution (`GraphInterrupt`/`AgentInterrupt`) and
//! writes an `OutputSchema(type=INTERACTION, index=…, payload=InteractionOutput)`
//! frame so a client can answer out of band. Rust keeps the wire frame and the
//! `InteractionOutput` payload shape, but models the wait explicitly: a
//! [`InteractionRequest`] is published through a [`StreamWriterManager`] and the
//! [`InteractionBroker`] awaits an [`InteractionResponse`] for it.
//!
//! # Contract
//!
//! - A request carries a minted id (`intr_<uuid hex>`), a [`InteractionKind`], the
//!   question payload, optional answer options, an optional timeout, and an
//!   optional [`CancellationToken`].
//! - [`InteractionRequest::to_stream_item`] renders the `__interaction__` output
//!   frame with payload `{"id": <component id or request id>, "value": <question>}`;
//!   the id falls back to the request id when no component id is set.
//! - [`InteractionDecision`] serializes with Python's approval vocabulary
//!   (`allow`, `allow_always`, `deny`, `cancel`, `answer`), not the Rust variant
//!   names, so frames stay wire compatible.
//! - [`InteractiveInput`] carries resume answers: `user_inputs` keyed by node id
//!   and `interrupt_answers` keyed by interrupt id. Seeded answers are consumed
//!   FIFO per key, and a seeded answer is returned *without* emitting a frame,
//!   mirroring Python's `_get_next_interactive_input`.
//! - Timeout and cancellation are additive to Python (which waits forever): the
//!   Rust runtime is cancellation-first, so both are surfaced as
//!   [`Error::Timeout`] / [`Error::Cancelled`].
//!
//! # Example
//!
//! Rendering the frame a client has to answer:
//!
//! ```
//! use openjiuwen_session::interaction::{InteractionRequest, INTERACTION_FRAME_TYPE};
//! use serde_json::json;
//!
//! let request = InteractionRequest::permission("tool.bash", json!({"command": "ls"}));
//! let frame = request.to_stream_item(0);
//! assert_eq!(frame.item_type, INTERACTION_FRAME_TYPE);
//! assert_eq!(
//!     frame.payload,
//!     json!({"id": "tool.bash", "value": {"command": "ls"}})
//! );
//! ```

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use openjiuwen_common::error::{Error, Result};

use crate::stream::{StreamItem, StreamWriterManager};

/// Frame type of a human-in-the-loop interaction output frame.
///
/// Python's `INTERACTION` constant; clients look for this `type` on output-mode
/// frames to find a request they must answer.
pub const INTERACTION_FRAME_TYPE: &str = "__interaction__";

/// State key under which [`InteractiveInput`] batches are stored.
///
/// Python's `INTERACTIVE_INPUT` constant, used when a run is resumed with
/// pre-seeded answers.
pub const INTERACTIVE_INPUT_KEY: &str = "__interactive_input__";

/// What a human is being asked to decide or provide.
///
/// The variants are the union of the interaction kinds used across the kernel;
/// the wire names are `snake_case` (`ask_user`, `skill_authorization`, …).
/// Defaults to [`InteractionKind::AskUser`] because a free-form question needs
/// no extra context, unlike a permission prompt.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InteractionKind {
    /// Approval to run a protected operation (tool call, command, …).
    Permission,
    /// A yes/no confirmation of a proposed action.
    Confirm,
    /// A free-form question for the user.
    #[default]
    AskUser,
    /// Authorization of a skill before it is loaded or run.
    SkillAuthorization,
    /// Approval of a skill evolution proposal.
    SkillEvolutionApproval,
    /// Approval of a tool call issued on behalf of a team member.
    TeamToolApproval,
    /// The client must execute a dynamically declared tool.
    ClientTool,
    /// Any other interaction a host application defines.
    Custom,
}

/// The answer to an [`InteractionRequest`].
///
/// Wire values follow Python's approval vocabulary (`allow`, `allow_always`,
/// `deny`) rather than the Rust variant names, so interaction responses stay
/// compatible with existing clients.
///
/// The default is [`InteractionDecision::Cancel`]: a malformed or empty response
/// never grants approval.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InteractionDecision {
    /// Approve this one request (`allow`).
    #[serde(rename = "allow")]
    Approve,
    /// Approve this request and every later one of the same kind
    /// (`allow_always`).
    #[serde(rename = "allow_always")]
    ApproveAlways,
    /// Reject the request (`deny`).
    #[serde(rename = "deny")]
    Reject,
    /// The request was abandoned without a decision (`cancel`).
    #[default]
    #[serde(rename = "cancel")]
    Cancel,
    /// The request was answered with data (`answer`).
    #[serde(rename = "answer")]
    Answer,
}

impl InteractionDecision {
    /// The `snake_case` wire name of this decision.
    ///
    /// Matches [`Serialize`] output; useful for logs and frame construction.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Approve => "allow",
            Self::ApproveAlways => "allow_always",
            Self::Reject => "deny",
            Self::Cancel => "cancel",
            Self::Answer => "answer",
        }
    }
}

/// One selectable answer of an [`InteractionRequest`].
///
/// Requests that present a choice list (UI prompts, tool approval menus) carry
/// these; `value` is the opaque payload echoed back inside
/// [`InteractionResponse::value`] when the option is chosen.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct InteractionOption {
    /// Stable option identifier.
    pub id: String,
    /// Human-readable label.
    pub label: String,
    /// Optional longer explanation.
    pub description: Option<String>,
    /// Payload sent back when this option is chosen.
    pub value: Value,
}

impl Default for InteractionOption {
    fn default() -> Self {
        Self {
            id: String::new(),
            label: String::new(),
            description: None,
            value: Value::Null,
        }
    }
}

/// A request for human input published on a session stream.
///
/// Created with [`InteractionRequest::new`] (or the [`InteractionRequest::permission`],
/// [`InteractionRequest::confirm`], and [`InteractionRequest::ask_user`]
/// shorthands), then handed to [`InteractionBroker::request`] together with a
/// [`StreamWriterManager`] so the frame reaches the client.
///
/// The cancellation token is not serialized: it is a runtime concern, so
/// [`InteractionRequest::with_cancellation`] is the only way to set it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct InteractionRequest {
    /// Request identifier, minted as `intr_<uuid hex>` when the request is built.
    pub id: String,
    /// What is being asked.
    pub kind: InteractionKind,
    /// Id of the component that asked, if any.
    ///
    /// When set, the emitted frame's payload `id` is this value instead of
    /// [`InteractionRequest::id`], matching Python's `executable_id` payload.
    pub component_id: Option<String>,
    /// The question payload handed to the client verbatim.
    pub question: Value,
    /// Selectable answers, if the client should offer a choice.
    pub options: Vec<InteractionOption>,
    /// Deadline in milliseconds; `None` falls back to the broker default.
    pub timeout_ms: Option<u64>,
    /// Free-form metadata for the host application.
    pub metadata: BTreeMap<String, Value>,
    /// Cancels the wait when tripped; never serialized.
    #[serde(skip)]
    pub cancellation: Option<CancellationToken>,
}

impl Default for InteractionRequest {
    fn default() -> Self {
        Self {
            id: String::new(),
            kind: InteractionKind::default(),
            component_id: None,
            question: Value::Null,
            options: Vec::new(),
            timeout_ms: None,
            metadata: BTreeMap::new(),
            cancellation: None,
        }
    }
}

impl InteractionRequest {
    /// Build a request with a freshly minted id and no deadline.
    #[must_use]
    pub fn new(kind: InteractionKind, question: Value) -> Self {
        Self {
            id: format!("intr_{}", openjiuwen_common::utils::id::uuid_hex()),
            kind,
            component_id: None,
            question,
            options: Vec::new(),
            timeout_ms: None,
            metadata: BTreeMap::new(),
            cancellation: None,
        }
    }

    /// Build a permission request for `component_id`.
    ///
    /// The request keeps its own minted id; `component_id` only changes the
    /// `id` field of the emitted frame payload.
    #[must_use]
    pub fn permission(component_id: impl Into<String>, question: Value) -> Self {
        Self::new(InteractionKind::Permission, question).with_component(component_id)
    }

    /// Build a yes/no confirmation request.
    #[must_use]
    pub fn confirm(question: Value) -> Self {
        Self::new(InteractionKind::Confirm, question)
    }

    /// Build a free-form question request.
    #[must_use]
    pub fn ask_user(question: Value) -> Self {
        Self::new(InteractionKind::AskUser, question)
    }

    /// Set the asking component's id.
    #[must_use]
    pub fn with_component(mut self, component_id: impl Into<String>) -> Self {
        self.component_id = Some(component_id.into());
        self
    }

    /// Set the deadline; sub-millisecond and unrepresentable values saturate at
    /// [`u64::MAX`] milliseconds.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout_ms = Some(u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX));
        self
    }

    /// Attach a cancellation token; tripping it aborts the wait.
    #[must_use]
    pub fn with_cancellation(mut self, token: CancellationToken) -> Self {
        self.cancellation = Some(token);
        self
    }

    /// Add one metadata entry.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Render the `__interaction__` output frame for this request.
    ///
    /// The payload is `{"id": <component id or request id>, "value": <question>}`,
    /// exactly Python's `InteractionOutput`.
    #[must_use]
    pub fn to_stream_item(&self, index: i64) -> StreamItem {
        let id = self.component_id.as_deref().unwrap_or(self.id.as_str());
        StreamItem::output(
            INTERACTION_FRAME_TYPE,
            Some(index),
            serde_json::json!({ "id": id, "value": self.question }),
        )
    }
}

/// The answer to an [`InteractionRequest`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct InteractionResponse {
    /// Id of the request this answers.
    pub request_id: String,
    /// What was decided.
    #[serde(default)]
    pub decision: InteractionDecision,
    /// Payload of the answer; [`Value::Null`] for plain approvals.
    pub value: Value,
    /// Optional human-readable explanation (set by rejections and cancellations).
    pub reason: Option<String>,
}

impl InteractionResponse {
    /// Approve the request once.
    #[must_use]
    pub fn approve(request_id: impl Into<String>) -> Self {
        Self::with_decision(request_id, InteractionDecision::Approve)
    }

    /// Approve the request and every later one of the same kind.
    #[must_use]
    pub fn approve_always(request_id: impl Into<String>) -> Self {
        Self::with_decision(request_id, InteractionDecision::ApproveAlways)
    }

    /// Reject the request with an optional reason.
    #[must_use]
    pub fn reject(request_id: impl Into<String>, reason: Option<String>) -> Self {
        Self {
            reason,
            ..Self::with_decision(request_id, InteractionDecision::Reject)
        }
    }

    /// Answer the request with `value`.
    #[must_use]
    pub fn answer(request_id: impl Into<String>, value: Value) -> Self {
        Self {
            value,
            ..Self::with_decision(request_id, InteractionDecision::Answer)
        }
    }

    /// Cancel the request (no decision was made).
    #[must_use]
    pub fn cancel(request_id: impl Into<String>) -> Self {
        Self::with_decision(request_id, InteractionDecision::Cancel)
    }

    /// Whether this response is [`InteractionDecision::Approve`] or
    /// [`InteractionDecision::ApproveAlways`].
    #[must_use]
    pub fn is_approved(&self) -> bool {
        matches!(
            self.decision,
            InteractionDecision::Approve | InteractionDecision::ApproveAlways
        )
    }

    /// Build a response with `decision` and a null value.
    fn with_decision(request_id: impl Into<String>, decision: InteractionDecision) -> Self {
        Self {
            request_id: request_id.into(),
            decision,
            value: Value::Null,
            reason: None,
        }
    }
}

/// Resume answers for a run, keyed by node id and interrupt id.
///
/// Ports Python's `InteractiveInput`: `user_inputs` holds answers addressed by
/// node, and `interrupt_answers` holds answers addressed by interrupt id.
/// [`InteractionBroker::seed`] queues both so a resumed run consumes them FIFO,
/// one per interrupted request.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct InteractiveInput {
    /// Answers keyed by node id.
    pub user_inputs: BTreeMap<String, Value>,
    /// The raw client payload, preserved verbatim for hosts that need it.
    pub raw_inputs: Option<Value>,
    /// Answers keyed by interrupt (interaction) id.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub interrupt_answers: BTreeMap<String, Value>,
}

impl InteractiveInput {
    /// An input with no answers.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an answer for `node_id`; an empty id is ignored.
    pub fn update(&mut self, node_id: impl Into<String>, value: Value) {
        let node_id = node_id.into();
        if node_id.is_empty() {
            tracing::debug!("ignoring interactive input for an empty node id");
            return;
        }
        self.user_inputs.insert(node_id, value);
    }

    /// Add an answer for `interrupt_id`; an empty id is ignored.
    pub fn answer(&mut self, interrupt_id: impl Into<String>, value: Value) {
        let interrupt_id = interrupt_id.into();
        if interrupt_id.is_empty() {
            tracing::debug!("ignoring interactive input for an empty interrupt id");
            return;
        }
        self.interrupt_answers.insert(interrupt_id, value);
    }

    /// Whether every lane is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.user_inputs.is_empty()
            && self.raw_inputs.is_none()
            && self.interrupt_answers.is_empty()
    }
}

/// Payload of an `__interaction__` frame.
///
/// Python's `InteractionOutput`: the `id` clients answer against and the
/// question `value`. [`InteractionRequest::to_stream_item`] builds this shape.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct InteractionOutput {
    /// Component id when the request named one, else the request id.
    pub id: String,
    /// The question payload.
    pub value: Value,
}

/// Broker linking interaction requests to the responses that answer them.
///
/// The broker owns the pending-request table and the seeded-answer queues, so it
/// is cheap to [`Clone`] (all state is behind an [`Arc`]) and safe to share
/// between the task waiting for input and the task delivering it.
///
/// # Seeded answers
///
/// [`InteractionBroker::seed`] and [`InteractionBroker::seed_interrupt`] queue
/// answers for a resumed run. A queued answer is handed out FIFO per request id,
/// and a request satisfied from the queue is *not* published on the stream —
/// Python's resume path never re-emits the frame either.
///
/// # Duplicate ids
///
/// Registering a second pending request with an id that is already waiting
/// overwrites the entry: the older sender is dropped, so its waiter observes a
/// closed channel ([`Error::Session`]) instead of silently receiving the answer
/// meant for the newer request. Ids are minted from UUIDs, so this only happens
/// when a caller pins ids by hand.
#[derive(Debug, Clone)]
pub struct InteractionBroker {
    inner: Arc<Inner>,
}

/// Shared broker state.
#[derive(Debug, Default)]
struct Inner {
    /// Waiters keyed by request id.
    pending: Mutex<HashMap<String, oneshot::Sender<InteractionResponse>>>,
    /// Seeded answers keyed by request/interrupt id, consumed FIFO.
    seeded: Mutex<HashMap<String, VecDeque<Value>>>,
    /// Fallback deadline for requests that do not carry one.
    default_timeout: Mutex<Option<Duration>>,
    /// Monotonic frame index for the emitted interaction frames.
    index: AtomicI64,
}

impl InteractionBroker {
    /// Create an empty broker with no default timeout.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Inner::default()),
        }
    }

    /// Set (or clear) the fallback deadline used by requests without their own.
    ///
    /// A `None` timeout means "wait forever", matching Python.
    pub fn set_default_timeout(&self, timeout: Option<Duration>) {
        *self.inner.default_timeout.lock() = timeout;
    }

    /// The fallback deadline, if one is configured.
    #[must_use]
    pub fn default_timeout(&self) -> Option<Duration> {
        *self.inner.default_timeout.lock()
    }

    /// Queue every answer of `input` as a seeded answer.
    ///
    /// `user_inputs` entries are queued under their key (the node id), so a
    /// request that uses the node id as its interaction id is answered, and
    /// `interrupt_answers` entries are queued under the interrupt id. Answers
    /// are consumed FIFO, one per matching request.
    pub fn seed(&self, input: InteractiveInput) {
        let mut seeded = self.inner.seeded.lock();
        let mut queued = 0_usize;
        for (key, value) in input.user_inputs {
            seeded.entry(key).or_default().push_back(value);
            queued += 1;
        }
        for (interrupt_id, value) in input.interrupt_answers {
            seeded.entry(interrupt_id).or_default().push_back(value);
            queued += 1;
        }
        drop(seeded);
        tracing::debug!(queued, "seeded interactive input answers");
    }

    /// Queue one answer for `interrupt_id`.
    pub fn seed_interrupt(&self, interrupt_id: impl Into<String>, value: Value) {
        let interrupt_id = interrupt_id.into();
        if interrupt_id.is_empty() {
            tracing::debug!("ignoring a seeded answer for an empty interrupt id");
            return;
        }
        self.inner
            .seeded
            .lock()
            .entry(interrupt_id)
            .or_default()
            .push_back(value);
    }

    /// The sorted ids of requests that are still waiting for a response.
    #[must_use]
    pub fn pending(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.inner.pending.lock().keys().cloned().collect();
        ids.sort();
        ids
    }

    /// Deliver `response` to the waiting request.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when no request is pending under
    /// `response.request_id`. A waiter that already gave up (timeout,
    /// cancellation, dropped future) is not an error: the response is dropped
    /// with a debug log.
    pub fn resolve(&self, response: InteractionResponse) -> Result<()> {
        let sender = self
            .inner
            .pending
            .lock()
            .remove(&response.request_id)
            .ok_or_else(|| Error::not_found(format!("interaction `{}`", response.request_id)))?;
        if sender.send(response).is_err() {
            tracing::debug!("interaction waiter dropped before the response arrived");
        }
        Ok(())
    }

    /// Cancel every pending request and drop all seeded answers.
    ///
    /// Each waiter receives an [`InteractionResponse::cancel`] carrying `reason`.
    pub fn cancel_all(&self, reason: impl Into<String>) {
        let reason = reason.into();
        let drained: Vec<(String, oneshot::Sender<InteractionResponse>)> = {
            let mut pending = self.inner.pending.lock();
            pending.drain().collect()
        };
        let cancelled = drained.len();
        for (request_id, sender) in drained {
            let response = InteractionResponse {
                request_id,
                decision: InteractionDecision::Cancel,
                value: Value::Null,
                reason: Some(reason.clone()),
            };
            if sender.send(response).is_err() {
                tracing::debug!("interaction waiter dropped before cancellation");
            }
        }
        self.inner.seeded.lock().clear();
        tracing::debug!(cancelled, "cancelled pending interactions");
    }

    /// Publish `request` and wait for the matching [`InteractionResponse`].
    ///
    /// The request is first checked against the seeded answers; a hit returns
    /// immediately and emits no frame. Otherwise the request is registered as
    /// pending, an `__interaction__` output frame is written to `writer` (its
    /// index comes from a broker-wide counter), and the broker waits until the
    /// request is resolved, times out, or is cancelled.
    ///
    /// The pending entry is removed on every exit path, so [`Self::pending`] is
    /// empty again once this future completes.
    ///
    /// # Errors
    ///
    /// - [`Error::Validation`] if the frame cannot be published (for example a
    ///   `Null` question payload).
    /// - [`Error::Timeout`] when the request's own deadline, or the broker
    ///   default, expires first.
    /// - [`Error::Cancelled`] when the request's [`CancellationToken`] is
    ///   cancelled.
    /// - [`Error::Session`] when the waiting channel closes without a response
    ///   (a duplicate id replaced this waiter, or the broker was dropped).
    pub async fn request(
        &self,
        request: InteractionRequest,
        writer: &StreamWriterManager,
    ) -> Result<InteractionResponse> {
        if let Some(value) = self.take_seeded(&request.id) {
            tracing::debug!(
                interaction_id = request.id.as_str(),
                "interaction answered from seeded input"
            );
            return Ok(InteractionResponse::answer(request.id, value));
        }

        let (sender, receiver) = oneshot::channel();
        {
            let mut pending = self.inner.pending.lock();
            if pending.insert(request.id.clone(), sender).is_some() {
                tracing::debug!(
                    interaction_id = request.id.as_str(),
                    "replaced a pending interaction with the same id"
                );
            }
        }

        let index = self.inner.index.fetch_add(1, Ordering::SeqCst);
        if let Err(error) = writer.write(request.to_stream_item(index)).await {
            self.remove_pending(&request.id);
            return Err(error);
        }

        let timeout = request
            .timeout_ms
            .map(Duration::from_millis)
            .or_else(|| self.default_timeout());
        let outcome = await_response(&request, receiver, timeout).await;
        self.remove_pending(&request.id);
        outcome
    }

    /// Pop the oldest seeded answer for `request_id`, if any.
    fn take_seeded(&self, request_id: &str) -> Option<Value> {
        let mut seeded = self.inner.seeded.lock();
        let queue = seeded.get_mut(request_id)?;
        let value = queue.pop_front();
        if queue.is_empty() {
            seeded.remove(request_id);
        }
        value
    }

    /// Drop a pending entry; missing entries are not an error.
    fn remove_pending(&self, request_id: &str) {
        if self.inner.pending.lock().remove(request_id).is_some() {
            tracing::debug!(
                interaction_id = request_id,
                "dropped an unanswered interaction registration"
            );
        }
    }
}

impl Default for InteractionBroker {
    fn default() -> Self {
        Self::new()
    }
}

/// Wait for `receiver`, honouring the cancellation token and the deadline.
///
/// Free function rather than a method: it needs no broker state, only the
/// request's cancellation token and timeout.
async fn await_response(
    request: &InteractionRequest,
    receiver: oneshot::Receiver<InteractionResponse>,
    timeout: Option<Duration>,
) -> Result<InteractionResponse> {
    let cancelled = request.cancellation.clone();
    let wait = async move {
        match cancelled {
            Some(token) => {
                tokio::select! {
                    biased;
                    () = token.cancelled() => Err(Error::cancelled()),
                    answer = receiver => answer
                        .map_err(|_| Error::session("interaction channel closed")),
                }
            }
            None => receiver
                .await
                .map_err(|_| Error::session("interaction channel closed")),
        }
    };

    match timeout {
        Some(limit) => match tokio::time::timeout(limit, wait).await {
            Ok(result) => result,
            Err(_elapsed) => {
                tracing::debug!(
                    interaction_id = request.id.as_str(),
                    ?limit,
                    "interaction timed out"
                );
                Err(Error::timeout(format!("interaction `{}`", request.id)))
            }
        },
        None => wait.await,
    }
}
