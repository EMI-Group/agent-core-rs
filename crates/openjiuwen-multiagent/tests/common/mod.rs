#![allow(dead_code)]
//! Shared fakes for the multi-agent integration suites.
//!
//! Every suite is its own crate, so helpers used by only one of them still
//! trigger dead-code warnings here; that is expected and silenced above.
//! Nothing in this module touches the network: "agents" are in-process closures
//! and streams are empty, so every suite stays offline and deterministic.
#![allow(unreachable_pub, clippy::unwrap_used, clippy::expect_used)]

use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_agent::{Agent, AgentCard, AgentChunk, AgentOutput};
use openjiuwen_common::error::Result;
use openjiuwen_session::Session;
use serde_json::Value;
use std::sync::Arc;

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
