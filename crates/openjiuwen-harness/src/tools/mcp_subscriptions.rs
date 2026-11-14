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
/// capacity and one [`SubscriptionState`] per subscribed URI. Without a provider every
/// `subscribe` answers [`NO_PROVIDER_REASON`]; `unsubscribe`, `list`, `poll` and `events` keep
/// working on subscriptions the host opened through [`Self::sink`] (out-of-band updates).
pub struct McpResourceSubscriptions {
    /// The provider asked to open subscriptions, when one is wired.
    provider: Option<Arc<dyn McpResourceSubscriptionProvider>>,
    /// Clamped per-subscription capacity.
    capacity: usize,
    /// Open subscriptions, ordered by URI.
    subscriptions: Mutex<BTreeMap<String, Arc<SubscriptionState>>>,
}

impl std::fmt::Debug for McpResourceSubscriptions {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("McpResourceSubscriptions")
            .field("capacity", &self.capacity)
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
    /// Builds a registry with [`DEFAULT_BUFFER_CAPACITY`] and no provider.
    #[must_use]
    pub fn new() -> Self {
        Self {
            provider: None,
            capacity: DEFAULT_BUFFER_CAPACITY,
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
    /// provider without `resources/subscribe` are [`Error::Unsupported`] with a reason, and a
    /// provider failure is forwarded unchanged (nothing is registered in that case).
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
        let state = Arc::new(SubscriptionState::new(uri, self.capacity));
        provider
            .subscribe(
                uri,
                &McpResourceSink {
                    state: Arc::clone(&state),
                },
            )
            .await?;
        self.subscriptions
            .lock()
            .insert(uri.to_owned(), Arc::clone(&state));
        Ok(state.info())
    }

    /// Closes the subscription for `uri`.
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
        let state =
            self.subscriptions.lock().get(uri).cloned().ok_or_else(|| {
                Error::not_found(format!("no MCP resource subscription for `{uri}`"))
            })?;
        if let Some(provider) = self.provider.clone() {
            provider.unsubscribe(uri).await?;
        }
        self.subscriptions.lock().remove(uri);
        Ok(state.info())
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
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when `uri` is not subscribed, and
    /// [`Error::Cancelled`](openjiuwen_common::error::Error::Cancelled) once the subscription was
    /// closed.
    pub async fn recv(&self, uri: &str) -> Result<McpResourceEvent> {
        let state = self.state(uri)?;
        let mut observer = state.observer.lock().await;
        loop {
            match observer.recv().await {
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
