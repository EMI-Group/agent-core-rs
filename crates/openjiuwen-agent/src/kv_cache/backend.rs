//! Deterministic in-process [`KvCacheActionProvider`] for tests and hosts
//! without a provider integration.
//!
//! Every action succeeds by default; [`InProcessKvCacheActionProvider`] records
//! each received action in call order, can be told to fail or reject a specific
//! action, and broadcasts every record to subscribers, so a test can observe the
//! action sequence a runtime produced without a network or a real provider.
use crate::kv_cache::actions::{
    KvCacheAction, KvCacheActionProvider, KvCacheActionRecord, KvCacheActionRequest,
};
use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::broadcast;

/// Capacity of the action broadcast channel.
///
/// A slow subscriber that falls further behind loses its oldest records; the
/// recording history is unaffected.
const ACTION_CHANNEL_CAPACITY: usize = 64;

/// Recorded actions, per-action outcomes, and the support flag.
#[derive(Debug)]
struct State {
    history: Vec<KvCacheActionRecord>,
    results: HashMap<KvCacheAction, bool>,
    errors: HashMap<KvCacheAction, String>,
    supported: bool,
}

impl Default for State {
    fn default() -> Self {
        Self {
            history: Vec::new(),
            results: HashMap::new(),
            errors: HashMap::new(),
            supported: true,
        }
    }
}

/// In-process action provider with a recorded, broadcast history.
#[derive(Debug, Clone)]
pub struct InProcessKvCacheActionProvider {
    state: Arc<Mutex<State>>,
    actions: broadcast::Sender<KvCacheActionRecord>,
}

impl Default for InProcessKvCacheActionProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl InProcessKvCacheActionProvider {
    /// Creates a provider whose every action succeeds.
    #[must_use]
    pub fn new() -> Self {
        let (actions, _receiver) = broadcast::channel(ACTION_CHANNEL_CAPACITY);
        Self {
            state: Arc::new(Mutex::new(State::default())),
            actions,
        }
    }

    /// Makes `action` report `succeed` instead of `true`.
    #[must_use]
    pub fn with_result(self, action: KvCacheAction, succeed: bool) -> Self {
        self.state.lock().results.insert(action, succeed);
        self
    }

    /// Makes `action` return an [`Error::transport`] with `message`.
    #[must_use]
    pub fn with_error(self, action: KvCacheAction, message: impl Into<String>) -> Self {
        self.state.lock().errors.insert(action, message.into());
        self
    }

    /// Makes [`KvCacheActionProvider::supports`] report `false`, so actions
    /// answer `Ok(false)` the way Python's model without a `*_kvc` method does.
    #[must_use]
    pub fn without_support(self) -> Self {
        self.state.lock().supported = false;
        self
    }

    /// Every action received so far, in call order.
    #[must_use]
    pub fn history(&self) -> Vec<KvCacheActionRecord> {
        self.state.lock().history.clone()
    }

    /// The actions received so far, in call order.
    #[must_use]
    pub fn actions(&self) -> Vec<KvCacheAction> {
        self.state
            .lock()
            .history
            .iter()
            .map(|record| record.action)
            .collect()
    }

    /// Number of received actions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.state.lock().history.len()
    }

    /// Whether no action has been received yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drops the recorded history; configured outcomes are kept.
    pub fn clear_history(&self) {
        self.state.lock().history.clear();
    }

    /// How many times `action` was received.
    #[must_use]
    pub fn calls(&self, action: KvCacheAction) -> usize {
        self.state
            .lock()
            .history
            .iter()
            .filter(|record| record.action == action)
            .count()
    }

    /// Subscribes to the action records received from now on.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<KvCacheActionRecord> {
        self.actions.subscribe()
    }

    /// Records `action`, broadcasts it, and resolves the configured outcome.
    fn record(&self, action: KvCacheAction, request: &KvCacheActionRequest) -> Result<bool> {
        let record = KvCacheActionRecord::new(action, request.clone());
        let (result, error) = {
            let mut state = self.state.lock();
            state.history.push(record.clone());
            (
                state.supported && state.results.get(&action).copied().unwrap_or(true),
                state.errors.get(&action).cloned(),
            )
        };
        // A broadcast with no live receivers is the normal case, not an error.
        let _ = self.actions.send(record);
        match error {
            Some(message) => Err(Error::transport(message)),
            None => Ok(result),
        }
    }
}

#[async_trait]
impl KvCacheActionProvider for InProcessKvCacheActionProvider {
    async fn prefetch(&self, request: &KvCacheActionRequest) -> Result<bool> {
        self.record(KvCacheAction::Prefetch, request)
    }

    async fn offload(&self, request: &KvCacheActionRequest) -> Result<bool> {
        self.record(KvCacheAction::Offload, request)
    }

    async fn evict(&self, request: &KvCacheActionRequest) -> Result<bool> {
        self.record(KvCacheAction::Evict, request)
    }

    fn supports(&self) -> bool {
        self.state.lock().supported
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn request() -> KvCacheActionRequest {
        KvCacheActionRequest::new("cache-a", "root-a").with_model("gpt-test")
    }

    #[tokio::test]
    async fn records_actions_in_call_order() {
        let provider = InProcessKvCacheActionProvider::new();
        assert!(provider.is_empty());
        assert_eq!(provider.len(), 0);

        assert!(provider
            .perform(KvCacheAction::Prefetch, &request())
            .await
            .expect("prefetch"));
        assert!(provider
            .perform(KvCacheAction::Offload, &request())
            .await
            .expect("offload"));
        assert_eq!(
            provider.actions(),
            [KvCacheAction::Prefetch, KvCacheAction::Offload]
        );
        assert_eq!(provider.len(), 2);
        assert_eq!(provider.calls(KvCacheAction::Prefetch), 1);
        assert_eq!(provider.calls(KvCacheAction::Evict), 0);
        assert_eq!(provider.history()[1].request, request());

        provider.clear_history();
        assert!(provider.is_empty());
    }

    #[tokio::test]
    async fn configured_outcomes_override_the_default() {
        let provider = InProcessKvCacheActionProvider::new()
            .with_result(KvCacheAction::Offload, false)
            .with_error(KvCacheAction::Evict, "provider unavailable");
        assert!(!provider
            .perform(KvCacheAction::Offload, &request())
            .await
            .expect("offload"));
        assert!(provider
            .perform(KvCacheAction::Prefetch, &request())
            .await
            .expect("prefetch"));
        let Err(error) = provider.perform(KvCacheAction::Evict, &request()).await else {
            panic!("evict must report the configured error");
        };
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Transport);
        assert_eq!(error.message(), "provider unavailable");
        // Errors are still recorded, so the call sequence stays observable.
        assert_eq!(provider.calls(KvCacheAction::Evict), 1);
    }

    #[tokio::test]
    async fn unsupported_provider_answers_false() {
        let provider = InProcessKvCacheActionProvider::new().without_support();
        assert!(!provider.supports());
        assert!(!provider
            .perform(KvCacheAction::Prefetch, &request())
            .await
            .expect("prefetch"));
        assert!(InProcessKvCacheActionProvider::new().supports());
    }

    #[tokio::test]
    async fn broadcast_subscribers_receive_every_action() {
        let provider = InProcessKvCacheActionProvider::new();
        let mut receiver = provider.subscribe();
        provider
            .perform(KvCacheAction::Prefetch, &request())
            .await
            .unwrap();
        provider
            .perform(KvCacheAction::Evict, &request())
            .await
            .unwrap();

        for expected in [KvCacheAction::Prefetch, KvCacheAction::Evict] {
            let record = tokio::time::timeout(Duration::from_secs(1), receiver.recv())
                .await
                .expect("record within the budget")
                .expect("record");
            assert_eq!(record.action, expected);
            assert_eq!(record.request.cache_id, "cache-a");
        }
    }

    #[tokio::test]
    async fn a_record_without_receivers_is_not_an_error() {
        let provider = InProcessKvCacheActionProvider::default();
        assert!(provider
            .perform(KvCacheAction::Evict, &request())
            .await
            .expect("evict"));
        assert_eq!(provider.calls(KvCacheAction::Evict), 1);
    }
}
