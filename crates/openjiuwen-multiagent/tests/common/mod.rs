#![allow(dead_code)]
//! Shared fakes for the multi-agent integration suites.
//!
//! Every suite is its own crate, so helpers used by only one of them still
//! trigger dead-code warnings here; that is expected and silenced above.
//! Nothing in this module touches the network: "agents" are in-process closures
//! and streams are empty, so every suite stays offline and deterministic.
#![allow(unreachable_pub, clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::new_ret_no_self)]

use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::{FutureExt, StreamExt};
use openjiuwen_agent::{Agent, AgentCard, AgentChunk, AgentOutput, Observation};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::tool::Tool;
use openjiuwen_multiagent::{
    AgentProvider, MessageEnvelope, MessageHandler, TeamAgent, ToolHost, HANDOFF_MESSAGE_KEY,
    HANDOFF_REASON_KEY, HANDOFF_TARGET_KEY,
};
use openjiuwen_session::Session;
use parking_lot::Mutex;
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};
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

/// An [`Agent`] whose handler may await, for delays and timeout cases.
///
/// [`FakeAgent`] takes a synchronous closure, so a suite that needs a slow
/// member (a handoff chain deadline, a blocked bus worker) uses this instead.
pub struct AsyncFakeAgent {
    card: AgentCard,
    handler: Box<AsyncFakeHandler>,
}

/// The asynchronous invocation handler of an [`AsyncFakeAgent`].
pub type AsyncFakeHandler = dyn Fn(Value, Option<Arc<dyn Session>>) -> BoxFuture<'static, Result<AgentOutput>>
    + Send
    + Sync;

impl AsyncFakeAgent {
    /// Builds a fake agent named by `card` that awaits `handler`.
    #[must_use]
    pub fn new(
        card: AgentCard,
        handler: impl Fn(Value, Option<Arc<dyn Session>>) -> BoxFuture<'static, Result<AgentOutput>>
            + Send
            + Sync
            + 'static,
    ) -> Arc<Self> {
        Arc::new(Self {
            card,
            handler: Box::new(handler),
        })
    }

    /// Builds an agent that sleeps `delay` and then answers `text`.
    #[must_use]
    pub fn delayed(name: &str, delay: Duration, text: &str) -> Arc<Self> {
        let card = agent_card(name);
        let text = text.to_owned();
        Self::new(card, move |_inputs, _session| {
            let text = text.clone();
            async move {
                tokio::time::sleep(delay).await;
                Ok(AgentOutput::answer(text))
            }
            .boxed()
        })
    }
}

#[async_trait::async_trait]
impl Agent for AsyncFakeAgent {
    fn card(&self) -> &AgentCard {
        &self.card
    }

    async fn invoke(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> Result<AgentOutput> {
        (self.handler)(inputs, session).await
    }

    fn stream(
        &self,
        _inputs: Value,
        _session: Option<Arc<dyn Session>>,
    ) -> BoxStream<'static, Result<AgentChunk>> {
        futures::stream::empty::<Result<AgentChunk>>().boxed()
    }
}

/// Builds an [`AgentOutput`] that carries a handoff directive.
///
/// The directive travels as a tool [`Observation`] because a plain
/// [`AgentOutput::answer`] has no top-level signal key; `None` message/reason
/// omit that key, and `message`/`reason` are forwarded verbatim.
#[must_use]
pub fn handoff_output(
    text: impl Into<String>,
    target: &str,
    message: Option<&str>,
    reason: Option<&str>,
) -> AgentOutput {
    let mut signal = serde_json::Map::new();
    signal.insert(
        HANDOFF_TARGET_KEY.to_owned(),
        Value::String(target.to_owned()),
    );
    if let Some(message) = message {
        signal.insert(
            HANDOFF_MESSAGE_KEY.to_owned(),
            Value::String(message.to_owned()),
        );
    }
    if let Some(reason) = reason {
        signal.insert(
            HANDOFF_REASON_KEY.to_owned(),
            Value::String(reason.to_owned()),
        );
    }
    AgentOutput {
        output: text.into(),
        observations: vec![Observation {
            content: Value::Object(signal).to_string(),
            ..Observation::default()
        }],
        ..AgentOutput::default()
    }
}

/// Builds a member provider that hands out `agent` per resolution.
#[must_use]
pub fn provider(agent: Arc<dyn Agent>) -> AgentProvider {
    Arc::new(move || TeamAgent::new(Arc::clone(&agent)))
}

/// Builds a member provider whose member also exposes `host` as its tool surface.
#[must_use]
pub fn host_provider(agent: Arc<dyn Agent>, host: Arc<dyn ToolHost>) -> AgentProvider {
    Arc::new(move || TeamAgent::with_tool_host(Arc::clone(&agent), Arc::clone(&host)))
}

/// Builds a provider that hands out `member` again and counts its resolutions.
///
/// Every call increments `calls`, so a suite can assert that a strategy resolved
/// an instance once and then reused the pinned one.
#[must_use]
pub fn counting_provider(member: TeamAgent, calls: Arc<AtomicUsize>) -> AgentProvider {
    Arc::new(move || {
        calls.fetch_add(1, Ordering::SeqCst);
        member.clone()
    })
}

/// A [`ToolHost`] that records the tools mounted on it.
///
/// Strategies inject tools without returning them to the caller, so a suite
/// observes the injection through this host: [`RecordingToolHost::mounts`] is the
/// mount log in call order, [`RecordingToolHost::card_count`] counts calls, and
/// [`RecordingToolHost::tool`] hands back a mounted tool so a suite can invoke it.
#[derive(Default)]
pub struct RecordingToolHost {
    /// Names of the mounted tool cards, in mount order.
    mounts: Mutex<Vec<String>>,
    /// The mounted tool handles, in mount order.
    tools: Mutex<Vec<Arc<dyn Tool>>>,
}

impl std::fmt::Debug for RecordingToolHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecordingToolHost")
            .field("mounts", &self.mounts.lock().clone())
            .finish_non_exhaustive()
    }
}

impl RecordingToolHost {
    /// Builds an empty recording host.
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Returns the mounted tool names, in mount order.
    #[must_use]
    pub fn mounts(&self) -> Vec<String> {
        self.mounts.lock().clone()
    }

    /// Returns how many `mount_tool` calls the host accepted.
    #[must_use]
    pub fn card_count(&self) -> usize {
        self.mounts.lock().len()
    }

    /// Returns the mounted tool handles, in mount order.
    #[must_use]
    pub fn tools(&self) -> Vec<Arc<dyn Tool>> {
        self.tools.lock().clone()
    }

    /// Returns the mounted tool whose card is named `name`, if any.
    #[must_use]
    pub fn tool(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools
            .lock()
            .iter()
            .find(|tool| tool.card().name == name)
            .cloned()
    }
}

impl ToolHost for RecordingToolHost {
    fn mount_tool(&self, tool: Arc<dyn Tool>) -> bool {
        self.mounts.lock().push(tool.card().name.clone());
        self.tools.lock().push(tool);
        true
    }
}

/// One frame a [`StreamAgent`] yields from [`Agent::stream`].
#[derive(Clone)]
pub enum StreamFrame {
    /// A regular agent chunk, forwarded verbatim.
    Chunk(AgentChunk),
    /// A failure that ends the stream; yielded as `Error::external(message)`.
    Fail(String),
}

/// An [`Agent`] with a scripted stream and a scripted answer.
///
/// [`FakeAgent`] streams nothing, so a suite exercising stream forwarding needs
/// frames. The agent records the inputs of every `stream` call, which is how a
/// suite observes `conversation_id`/`sender` shaping. Nothing touches the
/// network and no timer runs.
pub struct StreamAgent {
    card: AgentCard,
    frames: Vec<StreamFrame>,
    seen: Arc<Mutex<Vec<Value>>>,
}

impl StreamAgent {
    /// Builds an agent named by `card` that yields `frames` and answers `""`.
    #[must_use]
    pub fn new(card: AgentCard, frames: Vec<StreamFrame>) -> Arc<Self> {
        Arc::new(Self {
            card,
            frames,
            seen: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Builds an agent named `name` (`id == name`) that yields `frames`.
    #[must_use]
    pub fn streaming(name: &str, frames: Vec<StreamFrame>) -> Arc<Self> {
        Self::new(agent_card(name), frames)
    }

    /// Returns the inputs of every `stream` call, in call order.
    #[must_use]
    pub fn seen(&self) -> Vec<Value> {
        self.seen.lock().clone()
    }
}

#[async_trait::async_trait]
impl Agent for StreamAgent {
    fn card(&self) -> &AgentCard {
        &self.card
    }

    async fn invoke(
        &self,
        _inputs: Value,
        _session: Option<Arc<dyn Session>>,
    ) -> Result<AgentOutput> {
        Ok(AgentOutput::answer(""))
    }

    fn stream(
        &self,
        inputs: Value,
        _session: Option<Arc<dyn Session>>,
    ) -> BoxStream<'static, Result<AgentChunk>> {
        self.seen.lock().push(inputs);
        let frames = self.frames.clone();
        futures::stream::iter(frames.into_iter().map(|frame| match frame {
            StreamFrame::Chunk(chunk) => Ok(chunk),
            StreamFrame::Fail(message) => Err(Error::external(std::io::Error::other(message))),
        }))
        .boxed()
    }
}
