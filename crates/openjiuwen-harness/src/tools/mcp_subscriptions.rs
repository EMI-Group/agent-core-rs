//! MCP resource subscriptions: a bounded, lag-counted fan-out for `resources/updated`.
//!
//! [`super::mcp_resources`] covers the request/response half of the MCP resource surface
//! (`list_mcp_resources`, `read_mcp_resource`). This module covers the push half: a host that
//! can observe a server's resource-update notifications registers them with a
//! [`McpResourceSubscriptions`] registry, which keeps one bounded buffer per subscribed URI and
//! broadcasts every event to interested consumers.
//!
//! # Why this is a Rust extension
//!
//! Python's `openjiuwen/harness/tools/mcp_tools.py` only lists and reads resources: the resource
//! manager exposes no subscription surface, so there is no Python behaviour to port. The
//! registry here is the seam that makes `resources/subscribe` / `resources/updated` usable
//! without changing [`super::mcp_resources::McpResourceProvider`], and it is wired into the
//! model-facing surface through [`super::mcp_subscription_tool`] (`mcp_resource_subscription`).
//!
//! # Backwards compatibility
//!
//! [`McpResourceSubscriptionProvider`] is a **separate** trait, so every existing
//! `McpResourceProvider` implementation (including the fakes of upper layers) keeps compiling
//! untouched. Both of its methods are defaulted to [`Error::Unsupported`], so a provider that
//! only lists and reads resources can still be coerced into the subscription trait — subscribing
//! then answers with a typed reason naming the missing capability instead of a silent no-op.
//!
//! # Delivery model
//!
//! Each subscription owns two bounded channels, both sized by the registry capacity
//! ([`DEFAULT_BUFFER_CAPACITY`], clamped to [`MIN_BUFFER_CAPACITY`]..=[`MAX_BUFFER_CAPACITY`]):
//!
//! - a ring buffer drained by [`McpResourceSubscriptions::poll`] (the model-facing `poll`
//!   action), which drops the **oldest** event and increments the drop counter when full;
//! - a `tokio::sync::broadcast` channel for host consumers, observed through
//!   [`McpResourceSubscriptions::events`] (a fresh receiver) or
//!   [`McpResourceSubscriptions::recv`] (the registry's own long-lived receiver, which counts
//!   the events it missed as *lag* before returning the oldest still-buffered one).
//!
//! Publishing never blocks and never allocates an unbounded queue, so a provider may call
//! [`McpResourceSink::updated`] straight from a transport pump.
//!
//! # Bounded registry
//!
//! Each open subscription pins a ring buffer plus a broadcast channel, so the registry holds at
//! most [`DEFAULT_MAX_SUBSCRIPTIONS`] of them (overridable with
//! [`McpResourceSubscriptions::with_max_subscriptions`]). Subscribing one more answers
//! [`Error::Validation`]; [`McpResourceSubscriptions::unsubscribe`] and
//! [`McpResourceSubscriptions::reap`] free a slot again. A slot is never left behind for a
//! subscription the provider rejected or no longer knows: a failed open registers nothing, and an
//! `unsubscribe` the provider answers with [`Error::NotFound`] reaps the local entry instead of
//! pinning it for the lifetime of the registry.
// Size note: ~850 lines of implementation plus an inline test module; the registry,
// per-URI buffer and the model-facing tool share one file because they are one concern.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use openjiuwen_common::error::{Error, Result};

use super::mcp_resources::McpResourceContent;

/// Per-subscription buffer capacity used when the registry is not told otherwise.
pub const DEFAULT_BUFFER_CAPACITY: usize = 32;

/// Lower bound of the per-subscription buffer capacity.
pub const MIN_BUFFER_CAPACITY: usize = 1;

/// Upper bound of the per-subscription buffer capacity.
pub const MAX_BUFFER_CAPACITY: usize = 1024;

/// Number of concurrently open subscriptions a registry accepts when it is not told otherwise.
///
/// Every open subscription owns a ring buffer of [`DEFAULT_BUFFER_CAPACITY`] events plus a
/// broadcast channel, so the registry caps how many URIs it tracks instead of growing with the
/// model's input; [`McpResourceSubscriptions::with_max_subscriptions`] overrides the bound.
pub const DEFAULT_MAX_SUBSCRIPTIONS: usize = 64;

/// The reason a provider without `resources/subscribe` support reports.
///
/// Provider implementations may reuse it verbatim; it names the missing capability so a host
/// can tell "this server cannot subscribe" apart from a transport or permission failure.
pub const SUBSCRIBE_UNSUPPORTED: &str =
    "MCP resource subscriptions are not supported by this provider: \
     it exposes list_resources/read_resource but no resources/subscribe callback";

/// The reason [`McpResourceSubscriptionProvider::unsubscribe`]'s default reports.
pub const UNSUBSCRIBE_UNSUPPORTED: &str =
    "MCP resource unsubscription is not supported by this provider: \
     no subscription was ever opened by resources/subscribe";

/// The failure reported when a registry has no provider wired.
pub const NO_PROVIDER_REASON: &str =
    "mcp resource subscriptions are not configured: wire a provider implementing \
     McpResourceSubscriptionProvider (McpResourceSubscriptions::with_provider)";

/// One update notification of a subscribed MCP resource.
///
/// Python parity: the `notifications/resources/updated` payload (`uri` plus an optional
/// reason) with the resource content attached when the provider also read it, so a consumer can
/// act on one self-contained event. `sequence` is a per-subscription, 1-based counter assigned
/// by the registry, which makes gaps visible after a buffer overflow.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct McpResourceEvent {
    /// URI the notification is about.
    pub uri: String,
    /// 1-based position of the event within its subscription.
    pub sequence: u64,
    /// Server-provided reason for the update, when it sent one.
    pub reason: Option<String>,
    /// Resource content the provider attached, when it read the resource.
    pub content: Option<McpResourceContent>,
}

/// A snapshot of one subscription: identity, buffer occupancy and counters.
///
/// `buffered` is how many events the ring buffer still holds for
/// [`McpResourceSubscriptions::poll`]; `received` counts every published event, `dropped` the
/// ones the full ring buffer evicted, and `lagged` the ones the registry's own broadcast
/// receiver missed (see [`McpResourceSubscriptions::recv`]). `last_sequence` is the sequence of
/// the newest event, or `0` before the first one.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct McpSubscriptionInfo {
    /// The subscribed resource URI.
    pub uri: String,
    /// Buffer capacity of this subscription.
    pub capacity: usize,
    /// Events currently buffered and not yet polled.
    pub buffered: usize,
    /// Total events published into this subscription.
    pub received: u64,
    /// Events evicted from the full ring buffer.
    pub dropped: u64,
    /// Events the registry's broadcast receiver missed.
    pub lagged: u64,
    /// Sequence number of the newest event (`0` when none arrived yet).
    pub last_sequence: u64,
}

/// The mutable state of one subscription; shared by the registry and its [`McpResourceSink`].
#[derive(Debug)]
struct SubscriptionState {
    /// The subscribed URI.
    uri: String,
    /// Capacity of both the ring buffer and the broadcast channel.
    capacity: usize,
    /// Events waiting for [`McpResourceSubscriptions::poll`].
    buffered: Mutex<VecDeque<McpResourceEvent>>,
    /// Total events published.
    received: AtomicU64,
    /// Events evicted from the ring buffer.
    dropped: AtomicU64,
    /// Events the registry's own receiver missed.
    lagged: AtomicU64,
    /// Sequence of the newest event.
    sequence: AtomicU64,
    /// The fan-out channel handed to host consumers.
    sender: broadcast::Sender<McpResourceEvent>,
    /// The registry's long-lived receiver, used for [`McpResourceSubscriptions::recv`].
    observer: tokio::sync::Mutex<broadcast::Receiver<McpResourceEvent>>,
}

impl SubscriptionState {
    /// Creates the state of a fresh subscription for `uri`.
    fn new(uri: &str, capacity: usize) -> Self {
        let (sender, observer) = broadcast::channel(capacity);
        Self {
            uri: uri.to_owned(),
            capacity,
            buffered: Mutex::new(VecDeque::with_capacity(capacity)),
            received: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            lagged: AtomicU64::new(0),
            sequence: AtomicU64::new(0),
            sender,
            observer: tokio::sync::Mutex::new(observer),
        }
    }

    /// Assigns the next sequence number, buffers and broadcasts `event`.
    fn push(&self, mut event: McpResourceEvent) {
        let sequence = self.sequence.fetch_add(1, Ordering::SeqCst) + 1;
        event.sequence = sequence;
        event.uri.clone_from(&self.uri);
        self.received.fetch_add(1, Ordering::SeqCst);
        {
            let mut buffered = self.buffered.lock();
            if buffered.len() == self.capacity {
                buffered.pop_front();
                self.dropped.fetch_add(1, Ordering::SeqCst);
            }
            buffered.push_back(event.clone());
        }
        // An error only means "no live receiver", which is not a lost ring-buffer event: the
        // buffered copy above stays available to `poll`.
        let _ = self.sender.send(event);
    }

    /// Drains at most `limit` buffered events (`0` drains everything).
    fn drain(&self, limit: usize) -> Vec<McpResourceEvent> {
        let mut buffered = self.buffered.lock();
        let take = if limit == 0 {
            buffered.len()
        } else {
            limit.min(buffered.len())
        };
        buffered.drain(..take).collect()
    }

    /// The current snapshot.
    fn info(&self) -> McpSubscriptionInfo {
        McpSubscriptionInfo {
            uri: self.uri.clone(),
            capacity: self.capacity,
            buffered: self.buffered.lock().len(),
            received: self.received.load(Ordering::SeqCst),
            dropped: self.dropped.load(Ordering::SeqCst),
            lagged: self.lagged.load(Ordering::SeqCst),
            last_sequence: self.sequence.load(Ordering::SeqCst),
        }
    }
}

/// The publishing handle a [`McpResourceSubscriptionProvider`] writes updates into.
///
/// Cheap to clone (one `Arc`), `Send + Sync`, and bounded: publishing never awaits and never
/// grows without limit. A provider stores the sink next to its transport and calls
/// [`Self::updated`] when the server reports `notifications/resources/updated`.
#[derive(Debug, Clone)]
pub struct McpResourceSink {
    /// The subscription this sink publishes into.
    state: Arc<SubscriptionState>,
}

impl McpResourceSink {
    /// The URI this sink publishes for.
    #[must_use]
    pub fn uri(&self) -> &str {
        &self.state.uri
    }

    /// Publishes an update carrying the new content of the resource.
    pub fn updated(&self, content: McpResourceContent) {
        self.state.push(McpResourceEvent {
            uri: self.state.uri.clone(),
            sequence: 0,
            reason: None,
            content: Some(content),
        });
    }

    /// Publishes a bare notification (the server changed the resource without shipping it).
    pub fn notified(&self, reason: impl Into<String>) {
        self.state.push(McpResourceEvent {
            uri: self.state.uri.clone(),
            sequence: 0,
            reason: Some(reason.into()),
            content: None,
        });
    }

    /// The current snapshot of the subscription behind this sink.
    #[must_use]
    pub fn info(&self) -> McpSubscriptionInfo {
        self.state.info()
    }
}

/// The subscription half of the MCP resource seam.
///
/// Kept separate from [`super::mcp_resources::McpResourceProvider`] so existing providers keep
/// compiling. Both methods are defaulted to [`Error::Unsupported`]: a client that can list and
/// read but not subscribe answers with [`SUBSCRIBE_UNSUPPORTED`] instead of pretending to
/// subscribe.
#[async_trait]
pub trait McpResourceSubscriptionProvider: Send + Sync {
    /// Opens a subscription for `uri` and publishes every update into `sink`.
    ///
    /// The provider keeps the sink until [`Self::unsubscribe`] is called (or the client is
    /// dropped); it may publish from any task.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when the transport has no `resources/subscribe`
    /// capability, [`Error::Validation`] when the URI is not subscribable, and any transport
    /// error the server reports. A failed open leaves no subscription registered.
    async fn subscribe(&self, uri: &str, sink: &McpResourceSink) -> Result<()> {
        let _ = (uri, sink);
        Err(Error::unsupported(SUBSCRIBE_UNSUPPORTED))
    }

    /// Cancels the subscription for `uri`, releasing the sink the provider holds.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when the transport cannot subscribe at all, and any
    /// transport error the server reports (the local subscription is kept in that case).
    async fn unsubscribe(&self, uri: &str) -> Result<()> {
        let _ = uri;
        Err(Error::unsupported(UNSUBSCRIBE_UNSUPPORTED))
    }

    /// Whether this provider can open subscriptions at all.
    ///
    /// Defaults to `false`; a provider that overrides [`Self::subscribe`] reports `true`.
    fn supports_subscriptions(&self) -> bool {
        false
    }
}

/// The per-host registry of MCP resource subscriptions.
///
/// One instance per host (there is no process-global state): it holds the provider, the buffer
/// capacity, the subscription cap and one `SubscriptionState` per subscribed URI. Without a
/// provider every `subscribe` answers [`NO_PROVIDER_REASON`]; `unsubscribe`, `list`, `poll` and
/// `events` keep working on subscriptions the host opened through [`Self::sink`] (out-of-band
/// updates).
pub struct McpResourceSubscriptions {
    /// The provider asked to open subscriptions, when one is wired.
    provider: Option<Arc<dyn McpResourceSubscriptionProvider>>,
    /// Clamped per-subscription capacity.
    capacity: usize,
    /// Upper bound of the number of open subscriptions.
    max_subscriptions: usize,
    /// Open subscriptions, ordered by URI.
    subscriptions: Mutex<BTreeMap<String, Arc<SubscriptionState>>>,
}

impl std::fmt::Debug for McpResourceSubscriptions {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("McpResourceSubscriptions")
            .field("capacity", &self.capacity)
            .field("max_subscriptions", &self.max_subscriptions)
            .field("provider", &self.provider.is_some())
            .field(
                "subscriptions",
                &self.subscriptions.lock().keys().collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl Default for McpResourceSubscriptions {
    fn default() -> Self {
        Self::new()
    }
}

impl McpResourceSubscriptions {
    /// Builds a registry with [`DEFAULT_BUFFER_CAPACITY`], [`DEFAULT_MAX_SUBSCRIPTIONS`] and no
    /// provider.
    #[must_use]
    pub fn new() -> Self {
        Self {
            provider: None,
            capacity: DEFAULT_BUFFER_CAPACITY,
            max_subscriptions: DEFAULT_MAX_SUBSCRIPTIONS,
            subscriptions: Mutex::new(BTreeMap::new()),
        }
    }

    /// Wires the provider `subscribe`/`unsubscribe` delegate to.
    #[must_use]
    pub fn with_provider(mut self, provider: Arc<dyn McpResourceSubscriptionProvider>) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Overrides the per-subscription buffer capacity (clamped to
    /// [`MIN_BUFFER_CAPACITY`]..=[`MAX_BUFFER_CAPACITY`]).
    #[must_use]
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity.clamp(MIN_BUFFER_CAPACITY, MAX_BUFFER_CAPACITY);
        self
    }

    /// The configured per-subscription buffer capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Overrides the maximum number of concurrently open subscriptions (at least `1`).
    ///
    /// The default is [`DEFAULT_MAX_SUBSCRIPTIONS`]; subscribing past the bound answers
    /// [`Error::Validation`] without opening anything in the provider, so a model cannot pin an
    /// unbounded number of ring buffers.
    #[must_use]
    pub fn with_max_subscriptions(mut self, max_subscriptions: usize) -> Self {
        self.max_subscriptions = max_subscriptions.max(1);
        self
    }

    /// The configured maximum number of concurrently open subscriptions.
    #[must_use]
    pub fn max_subscriptions(&self) -> usize {
        self.max_subscriptions
    }

    /// The wired provider, when there is one.
    #[must_use]
    pub fn provider(&self) -> Option<&Arc<dyn McpResourceSubscriptionProvider>> {
        self.provider.as_ref()
    }

    /// Whether `uri` has an open subscription.
    #[must_use]
    pub fn is_subscribed(&self, uri: &str) -> bool {
        self.subscriptions.lock().contains_key(uri.trim())
    }

    /// Opens a subscription for `uri`.
    ///
    /// Idempotent: subscribing an already-subscribed URI returns its current snapshot without
    /// calling the provider again. Blank URIs are a validation error, a missing provider and a
    /// provider without `resources/subscribe` are [`Error::Unsupported`] with a reason, a
    /// provider failure is forwarded unchanged (nothing is registered in that case), and a
    /// subscription past [`Self::max_subscriptions`] is [`Error::Validation`] — a subscription
    /// the registry cannot track is never opened, and one it opened but lost a race for is
    /// undone in the provider.
    ///
    /// # Errors
    ///
    /// See above; the message of the error is what the model-facing tool renders.
    pub async fn subscribe(&self, uri: &str) -> Result<McpSubscriptionInfo> {
        let uri = uri.trim();
        if uri.is_empty() {
            return Err(Error::validation("uri is required"));
        }
        if let Some(state) = self.subscriptions.lock().get(uri).cloned() {
            return Ok(state.info());
        }
        let Some(provider) = self.provider.clone() else {
            return Err(Error::unsupported(NO_PROVIDER_REASON));
        };
        // Fast path: refuse an over-cap subscription before the provider round-trip.
        self.check_capacity(uri)?;
        let state = Arc::new(SubscriptionState::new(uri, self.capacity));
        provider
            .subscribe(
                uri,
                &McpResourceSink {
                    state: Arc::clone(&state),
                },
            )
            .await?;
        match self.register(uri, &state) {
            Ok(info) => Ok(info),
            Err(error) => {
                // A concurrent subscribe filled the last slot while the provider call was in
                // flight: undo the subscription the registry cannot track instead of pinning
                // its buffer for the lifetime of the registry.
                if let Err(undo) = provider.unsubscribe(uri).await {
                    tracing::debug!(
                        target: "openjiuwen_harness::tools::mcp_subscriptions",
                        uri,
                        error = %undo,
                        "could not undo an over-cap MCP resource subscription"
                    );
                }
                Err(error)
            }
        }
    }

    /// Closes the subscription for `uri`.
    ///
    /// When the provider reports [`Error::NotFound`] the subscription was already closed on the
    /// server side, so the local entry is reaped and its snapshot returned: the end state the
    /// caller asked for holds, and no ring buffer is pinned for a subscription nobody serves. Any
    /// other provider error keeps the subscription open.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when `uri` has no subscription, the provider's error when it
    /// cannot cancel (the local subscription then stays open), and [`Error::Validation`] for a
    /// blank URI.
    pub async fn unsubscribe(&self, uri: &str) -> Result<McpSubscriptionInfo> {
        let uri = uri.trim();
        if uri.is_empty() {
            return Err(Error::validation("uri is required"));
        }
        // Fail with `NotFound` before touching the provider when nothing is subscribed.
        let state = self.state(uri)?;
        if let Some(provider) = self.provider.clone() {
            if let Err(error) = provider.unsubscribe(uri).await {
                if !matches!(error, Error::NotFound(_)) {
                    return Err(error);
                }
                tracing::debug!(
                    target: "openjiuwen_harness::tools::mcp_subscriptions",
                    uri,
                    error = %error,
                    "reaping an MCP resource subscription the provider no longer knows"
                );
            }
        }
        self.subscriptions.lock().remove(uri);
        Ok(state.info())
    }

    /// Drops the local entry of `uri` without calling the provider, returning its last snapshot.
    ///
    /// For a host that saw the server close the subscription on its own side (a
    /// `notifications/resources/list_changed`, a transport reset) and for
    /// [`Self::unsubscribe`] when the provider no longer knows the subscription: the ring buffer
    /// is released as soon as the provider drops its [`McpResourceSink`], and the freed slot is
    /// available to the next `subscribe`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when `uri` is not subscribed and [`Error::Validation`] for a
    /// blank URI.
    pub fn reap(&self, uri: &str) -> Result<McpSubscriptionInfo> {
        let uri = uri.trim();
        if uri.is_empty() {
            return Err(Error::validation("uri is required"));
        }
        self.subscriptions
            .lock()
            .remove(uri)
            .map(|state| state.info())
            .ok_or_else(|| Error::not_found(format!("no MCP resource subscription for `{uri}`")))
    }

    /// Every open subscription, ordered by URI.
    #[must_use]
    pub fn list(&self) -> Vec<McpSubscriptionInfo> {
        self.subscriptions
            .lock()
            .values()
            .map(|state| state.info())
            .collect()
    }

    /// The snapshot of `uri`'s subscription.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when `uri` is not subscribed.
    pub fn info(&self, uri: &str) -> Result<McpSubscriptionInfo> {
        self.state(uri).map(|state| state.info())
    }

    /// Drains up to `limit` buffered events of `uri` (`0` drains everything).
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when `uri` is not subscribed.
    pub fn poll(&self, uri: &str, limit: usize) -> Result<Vec<McpResourceEvent>> {
        self.state(uri).map(|state| state.drain(limit))
    }

    /// Publishes an update into an already-open subscription, for a host that receives the
    /// server's notification outside the provider callback. Returns the assigned sequence.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when `uri` is not subscribed.
    pub fn publish(&self, uri: &str, content: McpResourceContent) -> Result<u64> {
        let state = self.state(uri)?;
        state.push(McpResourceEvent {
            uri: state.uri.clone(),
            sequence: 0,
            reason: None,
            content: Some(content),
        });
        Ok(state.sequence.load(Ordering::SeqCst))
    }

    /// A publishing handle for `uri`, for a host-owned notification pump.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when `uri` is not subscribed.
    pub fn sink(&self, uri: &str) -> Result<McpResourceSink> {
        self.state(uri).map(|state| McpResourceSink { state })
    }

    /// A fresh broadcast receiver of `uri`'s events, for a host consumer.
    ///
    /// A receiver created here only sees events published after the call; events it does not
    /// read fast enough surface as `broadcast` lag on its own side (only
    /// [`Self::recv`] feeds the registry's `lagged` counter).
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when `uri` is not subscribed.
    pub fn events(&self, uri: &str) -> Result<broadcast::Receiver<McpResourceEvent>> {
        self.state(uri).map(|state| state.sender.subscribe())
    }

    /// Receives the next event of `uri` through the registry's own long-lived receiver.
    ///
    /// Events the receiver missed while nobody called this method are counted as `lagged`; the
    /// next still-buffered event is returned. Awaiting here holds no lock on the subscription
    /// map, so publishing and polling stay available.
    ///
    /// Concurrent calls on the same URI do not queue behind each other: the caller that acquires
    /// the registry's shared receiver awaits it (keeping the lag accounting), and every other
    /// caller receives from its own [`broadcast::Receiver`] of the same sender — the view
    /// [`Self::events`] gives — so both observe the events published from then on. A per-call
    /// receiver only sees events published after it subscribed and never feeds `lagged`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when `uri` is not subscribed, and
    /// [`Error::Cancelled`] once the subscription was closed.
    pub async fn recv(&self, uri: &str) -> Result<McpResourceEvent> {
        let state = self.state(uri)?;
        recv_once(&state).await
    }

    /// Refuses a `subscribe` that would exceed [`Self::max_subscriptions`].
    fn check_capacity(&self, uri: &str) -> Result<()> {
        let open = self.subscriptions.lock().len();
        if open < self.max_subscriptions {
            return Ok(());
        }
        Err(self.capacity_error(uri, open))
    }

    /// Registers `state` under `uri`, or returns the entry a concurrent subscribe won with.
    fn register(&self, uri: &str, state: &Arc<SubscriptionState>) -> Result<McpSubscriptionInfo> {
        let mut subscriptions = self.subscriptions.lock();
        if let Some(existing) = subscriptions.get(uri) {
            return Ok(existing.info());
        }
        let open = subscriptions.len();
        if open >= self.max_subscriptions {
            // Never call `capacity_error` here: it reads the map this guard already holds.
            return Err(Self::capacity_message(uri, open, self.max_subscriptions));
        }
        subscriptions.insert(uri.to_owned(), Arc::clone(state));
        Ok(state.info())
    }

    /// The typed refusal of a `subscribe` past the subscription cap.
    fn capacity_error(&self, uri: &str, open: usize) -> Error {
        Self::capacity_message(uri, open, self.max_subscriptions)
    }

    /// The cap-refusal message for `open` of `max` slots in use.
    fn capacity_message(uri: &str, open: usize, max: usize) -> Error {
        Error::validation(format!(
            "MCP resource subscription limit reached ({open} of {max} open): unsubscribe a \
             resource before subscribing `{uri}`"
        ))
    }

    /// Looks one subscription up by (trimmed) URI.
    fn state(&self, uri: &str) -> Result<Arc<SubscriptionState>> {
        let uri = uri.trim();
        if uri.is_empty() {
            return Err(Error::validation("uri is required"));
        }
        self.subscriptions
            .lock()
            .get(uri)
            .cloned()
            .ok_or_else(|| Error::not_found(format!("no MCP resource subscription for `{uri}`")))
    }
}

/// Receives the next event of `state`, from the registry's shared receiver when it is free and
/// from a fresh receiver of the same sender otherwise.
///
/// The shared receiver keeps the lag accounting and only blocks *its own* caller: a concurrent
/// caller falls back to a per-call receiver (what [`McpResourceSubscriptions::events`] hands out)
/// instead of waiting behind the guard, so two `recv` calls on one URI both observe the events
/// published while they wait.
async fn recv_once(state: &SubscriptionState) -> Result<McpResourceEvent> {
    if let Ok(mut observer) = state.observer.try_lock() {
        recv_through(state, &mut observer).await
    } else {
        let mut receiver = state.sender.subscribe();
        recv_through(state, &mut receiver).await
    }
}

/// Receives the next event of `state` through `receiver`, counting its lag on `state`.
async fn recv_through(
    state: &SubscriptionState,
    receiver: &mut broadcast::Receiver<McpResourceEvent>,
) -> Result<McpResourceEvent> {
    loop {
        match receiver.recv().await {
            Ok(event) => return Ok(event),
            Err(broadcast::error::RecvError::Lagged(skipped)) => {
                state.lagged.fetch_add(skipped, Ordering::SeqCst);
                tracing::debug!(
                    target: "openjiuwen_harness::tools::mcp_subscriptions",
                    uri = state.uri,
                    skipped,
                    "MCP resource subscription receiver lagged"
                );
            }
            Err(broadcast::error::RecvError::Closed) => {
                return Err(Error::cancelled());
            }
        }
    }
}

/// A scriptable [`McpResourceSubscriptionProvider`] for tests and offline wiring.
///
/// It records every call, keeps the sink of each open subscription so a test can push updates,
/// and can be told to fail a specific URI or to behave like a client without
/// `resources/subscribe` support. Precedent:
/// [`super::mcp_resources::FakeMcpResourceProvider`].
#[derive(Debug, Default)]
pub struct FakeMcpResourceSubscriptionProvider {
    /// Sinks of the open subscriptions, keyed by URI.
    sinks: Mutex<BTreeMap<String, McpResourceSink>>,
    /// URIs subscribe was called with, in call order.
    subscribe_calls: Mutex<Vec<String>>,
    /// URIs unsubscribe was called with, in call order.
    unsubscribe_calls: Mutex<Vec<String>>,
    /// Per-URI subscribe failures.
    subscribe_failures: Mutex<BTreeMap<String, String>>,
    /// Per-URI unsubscribe failures.
    unsubscribe_failures: Mutex<BTreeMap<String, String>>,
    /// When set, every call reports [`SUBSCRIBE_UNSUPPORTED`].
    unsupported: bool,
}

impl FakeMcpResourceSubscriptionProvider {
    /// Builds a fake that accepts every well-formed subscription.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds a fake that behaves like a client without subscription support.
    #[must_use]
    pub fn unsupported() -> Self {
        Self {
            unsupported: true,
            ..Self::default()
        }
    }

    /// URIs passed to `subscribe`, in call order.
    #[must_use]
    pub fn subscribe_calls(&self) -> Vec<String> {
        self.subscribe_calls.lock().clone()
    }

    /// URIs passed to `unsubscribe`, in call order.
    #[must_use]
    pub fn unsubscribe_calls(&self) -> Vec<String> {
        self.unsubscribe_calls.lock().clone()
    }

    /// Whether `uri` currently has an open subscription.
    #[must_use]
    pub fn is_subscribed(&self, uri: &str) -> bool {
        self.sinks.lock().contains_key(uri)
    }

    /// Makes subscribing `uri` fail with `message`.
    pub fn fail_subscribe(&self, uri: &str, message: impl Into<String>) {
        self.subscribe_failures
            .lock()
            .insert(uri.to_owned(), message.into());
    }

    /// Makes unsubscribing `uri` fail with `message`.
    pub fn fail_unsubscribe(&self, uri: &str, message: impl Into<String>) {
        self.unsubscribe_failures
            .lock()
            .insert(uri.to_owned(), message.into());
    }

    /// Drops the server-side subscription of `uri` without telling the registry.
    ///
    /// Models a server that closed a subscription on its own (`resources/list_changed`, a
    /// transport reset): the registry still holds the entry, but [`Self::unsubscribe`] now answers
    /// [`Error::NotFound`]. Returns whether a subscription was open.
    pub fn drop_subscription(&self, uri: &str) -> bool {
        self.sinks.lock().remove(uri).is_some()
    }

    /// Pushes an update for `uri`; `false` when nothing is subscribed to it.
    pub fn push_content(&self, uri: &str, content: McpResourceContent) -> bool {
        let sink = self.sinks.lock().get(uri).cloned();
        match sink {
            Some(sink) => {
                sink.updated(content);
                true
            }
            None => false,
        }
    }

    /// Pushes a bare notification for `uri`; `false` when nothing is subscribed to it.
    pub fn push_notification(&self, uri: &str, reason: impl Into<String>) -> bool {
        let sink = self.sinks.lock().get(uri).cloned();
        match sink {
            Some(sink) => {
                sink.notified(reason);
                true
            }
            None => false,
        }
    }
}

#[async_trait]
impl McpResourceSubscriptionProvider for FakeMcpResourceSubscriptionProvider {
    async fn subscribe(&self, uri: &str, sink: &McpResourceSink) -> Result<()> {
        self.subscribe_calls.lock().push(uri.to_owned());
        if self.unsupported {
            return Err(Error::unsupported(SUBSCRIBE_UNSUPPORTED));
        }
        let failure = self.subscribe_failures.lock().get(uri).cloned();
        if let Some(message) = failure {
            return Err(Error::tool(message));
        }
        self.sinks.lock().insert(uri.to_owned(), sink.clone());
        Ok(())
    }

    async fn unsubscribe(&self, uri: &str) -> Result<()> {
        self.unsubscribe_calls.lock().push(uri.to_owned());
        if self.unsupported {
            return Err(Error::unsupported(UNSUBSCRIBE_UNSUPPORTED));
        }
        let failure = self.unsubscribe_failures.lock().get(uri).cloned();
        if let Some(message) = failure {
            return Err(Error::tool(message));
        }
        if self.sinks.lock().remove(uri).is_none() {
            return Err(Error::not_found(format!(
                "unknown MCP resource subscription: {uri}"
            )));
        }
        Ok(())
    }

    fn supports_subscriptions(&self) -> bool {
        !self.unsupported
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// Content with the given uri and text.
    fn content(uri: &str, text: &str) -> McpResourceContent {
        McpResourceContent {
            uri: uri.to_owned(),
            text: Some(text.to_owned()),
            ..McpResourceContent::default()
        }
    }

    /// A registry over `provider` with the default capacity.
    fn registry(provider: Arc<FakeMcpResourceSubscriptionProvider>) -> McpResourceSubscriptions {
        McpResourceSubscriptions::new().with_provider(provider)
    }

    #[tokio::test]
    async fn subscribe_is_idempotent_and_records_the_calls() {
        let provider = Arc::new(FakeMcpResourceSubscriptionProvider::new());
        let subscriptions = registry(provider.clone());

        let first = subscriptions.subscribe("mem://a").await.unwrap();
        assert_eq!(first.uri, "mem://a");
        assert_eq!(first.capacity, DEFAULT_BUFFER_CAPACITY);
        assert_eq!(first.received, 0);

        let second = subscriptions.subscribe("  mem://a  ").await.unwrap();
        assert_eq!(second, first);
        assert_eq!(provider.subscribe_calls(), vec!["mem://a".to_owned()]);
        assert!(subscriptions.is_subscribed("mem://a"));
        assert_eq!(subscriptions.list().len(), 1);
    }

    #[tokio::test]
    async fn subscribe_without_a_provider_or_without_support_is_typed_unsupported() {
        let bare = McpResourceSubscriptions::new();
        let error = bare.subscribe("mem://a").await.unwrap_err();
        assert!(matches!(error, Error::Unsupported(_)), "error: {error}");
        assert!(error.message().contains("with_provider"), "error: {error}");

        let provider = Arc::new(FakeMcpResourceSubscriptionProvider::unsupported());
        let subscriptions = registry(provider);
        let error = subscriptions.subscribe("mem://a").await.unwrap_err();
        assert!(matches!(error, Error::Unsupported(_)), "error: {error}");
        assert!(error.message().contains("resources/subscribe"));
        assert!(!subscriptions.is_subscribed("mem://a"));
        assert_eq!(subscriptions.list().len(), 0);
    }

    #[tokio::test]
    async fn blank_uri_and_provider_failures_are_reported() {
        let provider = Arc::new(FakeMcpResourceSubscriptionProvider::new());
        let subscriptions = registry(provider.clone());

        let error = subscriptions.subscribe("   ").await.unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "error: {error}");
        assert_eq!(error.message(), "uri is required");

        provider.fail_subscribe("mem://bad", "resource is not subscribable");
        let error = subscriptions.subscribe("mem://bad").await.unwrap_err();
        assert_eq!(error.message(), "resource is not subscribable");
        assert!(!subscriptions.is_subscribed("mem://bad"));
    }

    #[tokio::test]
    async fn published_events_are_buffered_and_drained() {
        let provider = Arc::new(FakeMcpResourceSubscriptionProvider::new());
        let subscriptions = registry(provider.clone());
        subscriptions.subscribe("mem://a").await.unwrap();

        assert!(provider.push_content("mem://a", content("mem://a", "one")));
        assert!(provider.push_notification("mem://a", "touched"));
        assert!(!provider.push_content("mem://b", content("mem://b", "x")));

        let info = subscriptions.info("mem://a").unwrap();
        assert_eq!(info.received, 2);
        assert_eq!(info.buffered, 2);
        assert_eq!(info.last_sequence, 2);
        assert_eq!(info.dropped, 0);

        let events = subscriptions.poll("mem://a", 1).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].sequence, 1);
        assert_eq!(
            events[0].content.as_ref().unwrap().text.as_deref(),
            Some("one")
        );
        assert_eq!(events[0].uri, "mem://a");

        let rest = subscriptions.poll("mem://a", 0).unwrap();
        assert_eq!(rest.len(), 1);
        assert_eq!(rest[0].reason.as_deref(), Some("touched"));
        assert_eq!(subscriptions.poll("mem://a", 10).unwrap().len(), 0);
        assert_eq!(subscriptions.info("mem://a").unwrap().buffered, 0);
    }

    #[tokio::test]
    async fn the_ring_buffer_drops_the_oldest_event_and_counts_it() {
        let provider = Arc::new(FakeMcpResourceSubscriptionProvider::new());
        let subscriptions = registry(provider.clone()).with_capacity(2);
        subscriptions.subscribe("mem://a").await.unwrap();

        for text in ["one", "two", "three"] {
            assert!(provider.push_content("mem://a", content("mem://a", text)));
        }

        let info = subscriptions.info("mem://a").unwrap();
        assert_eq!(info.capacity, 2);
        assert_eq!(info.received, 3);
        assert_eq!(info.dropped, 1);
        assert_eq!(info.buffered, 2);

        let texts: Vec<Option<String>> = subscriptions
            .poll("mem://a", 0)
            .unwrap()
            .into_iter()
            .map(|event| event.content.and_then(|content| content.text))
            .collect();
        assert_eq!(
            texts,
            vec![Some("two".to_owned()), Some("three".to_owned())]
        );
    }

    #[test]
    fn capacity_is_clamped() {
        assert_eq!(
            McpResourceSubscriptions::new().capacity(),
            DEFAULT_BUFFER_CAPACITY
        );
        assert_eq!(
            McpResourceSubscriptions::new().with_capacity(0).capacity(),
            MIN_BUFFER_CAPACITY
        );
        assert_eq!(
            McpResourceSubscriptions::new()
                .with_capacity(10_000)
                .capacity(),
            MAX_BUFFER_CAPACITY
        );
    }

    #[tokio::test]
    async fn recv_counts_lag_and_returns_the_oldest_surviving_event() {
        let provider = Arc::new(FakeMcpResourceSubscriptionProvider::new());
        let subscriptions = registry(provider.clone()).with_capacity(2);
        subscriptions.subscribe("mem://a").await.unwrap();

        for text in ["one", "two", "three", "four"] {
            assert!(provider.push_content("mem://a", content("mem://a", text)));
        }

        let event = subscriptions.recv("mem://a").await.unwrap();
        assert_eq!(event.sequence, 3);
        assert_eq!(subscriptions.info("mem://a").unwrap().lagged, 2);
        assert_eq!(
            subscriptions
                .recv("mem://a")
                .await
                .unwrap()
                .content
                .unwrap()
                .text
                .as_deref(),
            Some("four")
        );
    }

    #[tokio::test]
    async fn concurrent_recv_calls_over_the_same_uri_both_observe_events() {
        let provider = Arc::new(FakeMcpResourceSubscriptionProvider::new());
        let subscriptions = Arc::new(registry(provider.clone()));
        subscriptions.subscribe("mem://a").await.unwrap();

        let first = tokio::spawn({
            let subscriptions = Arc::clone(&subscriptions);
            async move { subscriptions.recv("mem://a").await }
        });
        let second = tokio::spawn({
            let subscriptions = Arc::clone(&subscriptions);
            async move { subscriptions.recv("mem://a").await }
        });
        // Let both tasks park on their receiver: one takes the registry's shared receiver, the
        // other fans out through its own instead of queueing behind that guard.
        for _ in 0..4 {
            tokio::task::yield_now().await;
        }
        assert!(provider.push_notification("mem://a", "changed"));

        let (one, two) = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            (
                first.await.unwrap().unwrap(),
                second.await.unwrap().unwrap(),
            )
        })
        .await
        .expect("two concurrent recv calls must not deadlock");
        for event in [one, two] {
            assert_eq!(event.sequence, 1);
            assert_eq!(event.reason.as_deref(), Some("changed"));
        }
    }

    #[test]
    fn the_subscription_cap_has_a_default_and_is_configurable() {
        assert_eq!(
            McpResourceSubscriptions::new().max_subscriptions(),
            DEFAULT_MAX_SUBSCRIPTIONS
        );
        assert_eq!(
            McpResourceSubscriptions::new()
                .with_max_subscriptions(3)
                .max_subscriptions(),
            3
        );
        // A registry always accepts at least one subscription.
        assert_eq!(
            McpResourceSubscriptions::new()
                .with_max_subscriptions(0)
                .max_subscriptions(),
            1
        );
    }

    #[tokio::test]
    async fn subscribing_past_the_cap_is_a_validation_error_that_frees_slot_on_unsubscribe() {
        let provider = Arc::new(FakeMcpResourceSubscriptionProvider::new());
        let subscriptions = registry(provider.clone()).with_max_subscriptions(2);
        subscriptions.subscribe("mem://a").await.unwrap();
        subscriptions.subscribe("mem://b").await.unwrap();

        let error = subscriptions.subscribe("mem://c").await.unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "error: {error}");
        assert!(error.message().contains("limit reached"), "error: {error}");
        assert!(error.message().contains("2 of 2"), "error: {error}");
        assert!(!subscriptions.is_subscribed("mem://c"));
        // Refused before the provider round-trip: nothing was opened and nothing registered.
        assert_eq!(
            provider.subscribe_calls(),
            vec!["mem://a".to_owned(), "mem://b".to_owned()]
        );
        assert_eq!(subscriptions.list().len(), 2);

        // Re-subscribing an already-open URI stays idempotent at the cap.
        assert_eq!(
            subscriptions.subscribe("mem://a").await.unwrap().uri,
            "mem://a"
        );

        subscriptions.unsubscribe("mem://a").await.unwrap();
        assert_eq!(
            subscriptions.subscribe("mem://c").await.unwrap().uri,
            "mem://c"
        );
    }

    #[tokio::test]
    async fn unsubscribe_reaps_an_entry_the_provider_no_longer_knows() {
        let provider = Arc::new(FakeMcpResourceSubscriptionProvider::new());
        let subscriptions = registry(provider.clone()).with_max_subscriptions(1);
        subscriptions.subscribe("mem://a").await.unwrap();
        // The server closed the subscription on its own side.
        assert!(provider.drop_subscription("mem://a"));

        let reaped = subscriptions.unsubscribe("mem://a").await.unwrap();
        assert_eq!(reaped.uri, "mem://a");
        assert!(!subscriptions.is_subscribed("mem://a"));
        assert!(subscriptions.info("mem://a").is_err());
        // The reaped slot is free again: the cap must not count a subscription nobody serves.
        assert_eq!(
            subscriptions.subscribe("mem://b").await.unwrap().uri,
            "mem://b"
        );
    }

    #[tokio::test]
    async fn reap_drops_one_entry_without_calling_the_provider() {
        let provider = Arc::new(FakeMcpResourceSubscriptionProvider::new());
        let subscriptions = registry(provider.clone());
        subscriptions.subscribe("mem://a").await.unwrap();
        assert!(provider.push_notification("mem://a", "changed"));

        let snapshot = subscriptions.reap("mem://a").unwrap();
        assert_eq!(snapshot.uri, "mem://a");
        assert_eq!(snapshot.received, 1);
        assert!(!subscriptions.is_subscribed("mem://a"));
        assert_eq!(provider.unsubscribe_calls(), Vec::<String>::new());
        assert!(matches!(
            subscriptions.reap("mem://a").unwrap_err(),
            Error::NotFound(_)
        ));
        assert!(matches!(
            subscriptions.reap("  ").unwrap_err(),
            Error::Validation(_)
        ));
    }

    #[tokio::test]
    async fn events_gives_a_fresh_broadcast_receiver() {
        let provider = Arc::new(FakeMcpResourceSubscriptionProvider::new());
        let subscriptions = registry(provider.clone());
        subscriptions.subscribe("mem://a").await.unwrap();
        let mut receiver = subscriptions.events("mem://a").unwrap();

        assert!(provider.push_notification("mem://a", "changed"));
        let event = receiver.recv().await.unwrap();
        assert_eq!(event.reason.as_deref(), Some("changed"));
        assert_eq!(event.sequence, 1);
    }

    #[tokio::test]
    async fn unsubscribe_removes_the_subscription_and_forwards_failures() {
        let provider = Arc::new(FakeMcpResourceSubscriptionProvider::new());
        let subscriptions = registry(provider.clone());
        subscriptions.subscribe("mem://a").await.unwrap();

        let closed = subscriptions.unsubscribe("mem://a").await.unwrap();
        assert_eq!(closed.uri, "mem://a");
        assert_eq!(provider.unsubscribe_calls(), vec!["mem://a".to_owned()]);
        assert!(!subscriptions.is_subscribed("mem://a"));
        assert!(!provider.is_subscribed("mem://a"));

        let error = subscriptions.unsubscribe("mem://a").await.unwrap_err();
        assert!(matches!(error, Error::NotFound(_)), "error: {error}");

        subscriptions.subscribe("mem://b").await.unwrap();
        provider.fail_unsubscribe("mem://b", "server is gone");
        let error = subscriptions.unsubscribe("mem://b").await.unwrap_err();
        assert_eq!(error.message(), "server is gone");
        assert!(subscriptions.is_subscribed("mem://b"));
    }

    #[tokio::test]
    async fn unknown_uris_are_not_found() {
        let subscriptions = McpResourceSubscriptions::new();
        for error in [
            subscriptions.info("mem://a").unwrap_err(),
            subscriptions.poll("mem://a", 1).unwrap_err(),
            subscriptions
                .publish("mem://a", content("mem://a", "x"))
                .unwrap_err(),
            subscriptions.sink("mem://a").unwrap_err(),
            subscriptions.events("mem://a").unwrap_err(),
        ] {
            assert!(matches!(error, Error::NotFound(_)), "error: {error}");
        }
        let error = subscriptions.info("  ").unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "error: {error}");
    }

    #[tokio::test]
    async fn publish_and_sink_publish_into_an_open_subscription() {
        let subscriptions = McpResourceSubscriptions::new().with_capacity(4);
        // Without a provider the subscribe fails, which is what leaves the registry empty here.
        let error = subscriptions.subscribe("mem://a").await.err().unwrap();
        assert!(matches!(error, Error::Unsupported(_)), "error: {error}");

        // Registered out of band: a host that opened the subscription itself.
        let state = Arc::new(SubscriptionState::new("mem://a", 4));
        subscriptions
            .subscriptions
            .lock()
            .insert("mem://a".to_owned(), Arc::clone(&state));

        assert_eq!(
            subscriptions
                .publish("mem://a", content("mem://a", "a"))
                .unwrap(),
            1
        );
        let sink = subscriptions.sink("mem://a").unwrap();
        assert_eq!(sink.uri(), "mem://a");
        sink.notified("touched");
        assert_eq!(sink.info().received, 2);
        assert_eq!(subscriptions.poll("mem://a", 0).unwrap().len(), 2);
    }

    #[test]
    fn events_and_info_serialize_snake_case() {
        let event = McpResourceEvent {
            uri: "mem://a".to_owned(),
            sequence: 7,
            reason: Some("changed".to_owned()),
            content: None,
        };
        let rendered = serde_json::to_value(&event).unwrap();
        assert_eq!(rendered["sequence"], 7);
        assert_eq!(rendered["reason"], "changed");
        assert_eq!(rendered["content"], serde_json::Value::Null);

        let partial: McpResourceEvent =
            serde_json::from_value(serde_json::json!({"uri": "mem://a", "extra": 1})).unwrap();
        assert_eq!(partial.sequence, 0);
        assert_eq!(partial.uri, "mem://a");

        let info = McpSubscriptionInfo::default();
        assert_eq!(serde_json::to_value(&info).unwrap()["last_sequence"], 0);
    }

    #[test]
    fn fake_provider_reports_its_support() {
        assert!(FakeMcpResourceSubscriptionProvider::new().supports_subscriptions());
        assert!(!FakeMcpResourceSubscriptionProvider::unsupported().supports_subscriptions());
        assert_eq!(
            FakeMcpResourceSubscriptionProvider::default().subscribe_calls(),
            Vec::<String>::new()
        );
    }
}
