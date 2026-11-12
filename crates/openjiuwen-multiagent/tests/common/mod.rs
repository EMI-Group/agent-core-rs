#![allow(dead_code)]
//! Shared fakes for the multi-agent integration suites.
//!
//! Every suite is its own crate, so helpers used by only one of them still
//! trigger dead-code warnings here; that is expected and silenced above.
//! Nothing in this module touches the network: "agents" are in-process closures
//! and streams are empty, so every suite stays offline and deterministic.
#![allow(unreachable_pub, clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::new_ret_no_self)]

use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_agent::{Agent, AgentCard, AgentChunk, AgentOutput};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_multiagent::{MessageEnvelope, MessageHandler};
use openjiuwen_session::Session;
use parking_lot::Mutex;
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

/// Builds a card whose id and name are both `name`.
pub fn agent_card(name: &str) -> AgentCard {
    AgentCard::new(name).with_id(name)
}

/// The invocation handler of a [`FakeAgent`].
pub type FakeHandler = dyn Fn(Value, Option<Arc<dyn Session>>) -> Result<AgentOutput> + Send + Sync;

/// An [`Agent`] backed by a plain closure.
///
/// The closure receives the invoke inputs and the optional session and returns
/// the output verbatim, so a suite can script member behaviour without a model.
pub struct FakeAgent {
    card: AgentCard,
    handler: Box<FakeHandler>,
}

impl FakeAgent {
    /// Builds a fake agent named by `card` that answers through `handler`.
    #[must_use]
    pub fn new(
        card: AgentCard,
        handler: impl Fn(Value, Option<Arc<dyn Session>>) -> Result<AgentOutput> + Send + Sync + 'static,
    ) -> Arc<Self> {
        Arc::new(Self {
            card,
            handler: Box::new(handler),
        })
    }

    /// Builds a fake agent that answers `"{name}:{text}"`.
    ///
    /// `text` is the string `inputs["query"]` when present, else the compact
    /// JSON rendering of the inputs.
    #[must_use]
    pub fn echoing(name: &str) -> Arc<Self> {
        let card = agent_card(name);
        let label = name.to_owned();
        Self::new(card, move |inputs, _session| {
            let text = inputs
                .get("query")
                .and_then(Value::as_str)
                .map_or_else(|| inputs.to_string(), ToOwned::to_owned);
            Ok(AgentOutput::answer(format!("{label}:{text}")))
        })
    }
}

#[async_trait::async_trait]
impl Agent for FakeAgent {
    fn card(&self) -> &AgentCard {
        &self.card
    }

    async fn invoke(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> Result<AgentOutput> {
        (self.handler)(inputs, session)
    }

    fn stream(
        &self,
        _inputs: Value,
        _session: Option<Arc<dyn Session>>,
    ) -> BoxStream<'static, Result<AgentChunk>> {
        futures::stream::empty::<Result<AgentChunk>>().boxed()
    }
}

/// A [`MessageHandler`] that records every delivery and scripts its replies.
///
/// Used by the message-bus suites: it keeps the P2P payloads and the pub/sub
/// delivery order, answers P2P with a configurable value, can delay its first
/// P2P delivery (to observe FIFO ordering), and can fail selected subscribers.
#[derive(Debug, Default)]
pub struct RecordingHandler {
    /// Every P2P envelope handled, in handling order.
    pub p2p: Mutex<Vec<MessageEnvelope>>,
    /// Every pub/sub delivery as `(subscriber, envelope)`, in handling order.
    pub pubsub: Mutex<Vec<(String, MessageEnvelope)>>,
    /// The reply returned by every P2P delivery; `json!({"ok": true})` initially.
    pub p2p_reply: Mutex<Value>,
    /// Delay slept through before recording the first P2P delivery only.
    pub first_delay: Mutex<Option<Duration>>,
    /// Subscriber ids whose pub/sub delivery fails.
    pub fail_subscribers: Mutex<BTreeSet<String>>,
}

impl RecordingHandler {
    /// Builds a handler with the default `{"ok": true}` P2P reply.
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            p2p_reply: Mutex::new(json!({"ok": true})),
            ..Self::default()
        })
    }

    /// Sets the value every P2P delivery replies with.
    pub fn set_reply(&self, value: Value) {
        *self.p2p_reply.lock() = value;
    }

    /// Delays the first P2P delivery by `delay`.
    pub fn delay_first(&self, delay: Duration) {
        *self.first_delay.lock() = Some(delay);
    }

    /// Makes pub/sub delivery to `subscriber` fail.
    pub fn fail_for(&self, subscriber: &str) {
        self.fail_subscribers.lock().insert(subscriber.to_owned());
    }

    /// Returns the recorded P2P payloads, in handling order.
    #[must_use]
    pub fn p2p_payloads(&self) -> Vec<Value> {
        self.p2p
            .lock()
            .iter()
            .map(|envelope| envelope.message.clone())
            .collect()
    }

    /// Returns the recorded pub/sub subscriber ids, in handling order.
    #[must_use]
    pub fn pubsub_order(&self) -> Vec<String> {
        self.pubsub
            .lock()
            .iter()
            .map(|(subscriber, _envelope)| subscriber.clone())
            .collect()
    }
}

#[async_trait::async_trait]
impl MessageHandler for RecordingHandler {
    async fn handle_p2p(&self, envelope: &MessageEnvelope) -> Result<Value> {
        let delay = self.first_delay.lock().take();
        if let Some(delay) = delay {
            tokio::time::sleep(delay).await;
        }
        self.p2p.lock().push(envelope.clone());
        Ok(self.p2p_reply.lock().clone())
    }

    async fn handle_pubsub(&self, envelope: &MessageEnvelope, subscriber: &str) -> Result<()> {
        if self.fail_subscribers.lock().contains(subscriber) {
            return Err(Error::transport(format!(
                "subscriber '{subscriber}' failed"
            )));
        }
        self.pubsub
            .lock()
            .push((subscriber.to_owned(), envelope.clone()));
        Ok(())
    }
}
