//! The in-process transport: 1:1 delegation to the crate's [`MessageBus`].
//!
//! Nothing is re-implemented here. Topic naming, the per-topic FIFO workers,
//! request/response correlation, deadlines, and pub/sub fan-out all stay in
//! [`MessageBus`], so a team routed through [`InProcessTransport`] behaves
//! exactly like a team talking to the bus directly.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_common::error::Result;
use serde_json::Value;

use crate::envelope::MessageEnvelope;
use crate::message_bus::{MessageBus, MessageBusConfig, MessageHandler};

use super::{TeamTransport, TransportKind};

/// A [`TeamTransport`] backed by the in-process [`MessageBus`].
pub struct InProcessTransport {
    /// The bus every call is delegated to.
    bus: Arc<MessageBus>,
}

impl InProcessTransport {
    /// Builds a stopped transport for `team_id`.
    ///
    /// `team_id` is stamped onto `bus_config`, so the transport's team id and
    /// the bus's transport topic names always agree (an empty id falls back to
    /// `"default"`, matching [`MessageBus`]).
    #[must_use]
    pub fn new(
        team_id: impl Into<String>,
        bus_config: MessageBusConfig,
        handler: Arc<dyn MessageHandler>,
    ) -> Arc<Self> {
        let config = bus_config.with_team_id(team_id.into());
        Arc::new(Self {
            bus: Arc::new(MessageBus::new(config, handler)),
        })
    }

    /// Wraps an already-built `bus` without creating another one.
    ///
    /// This is how a [`TeamRuntime`](crate::TeamRuntime) exposes its own bus as
    /// the default transport: outbound deliveries go through the seam while the
    /// bus keeps owning the topic workers and the inbound dispatch.
    #[must_use]
    pub fn from_bus(bus: Arc<MessageBus>) -> Arc<Self> {
        Arc::new(Self { bus })
    }

    /// Returns the bus this transport delegates to.
    #[must_use]
    pub fn bus(&self) -> Arc<MessageBus> {
        Arc::clone(&self.bus)
    }
}

impl fmt::Debug for InProcessTransport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("InProcessTransport")
            .field("team_id", &self.bus.team_id())
            .field("started", &self.bus.is_started())
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl TeamTransport for InProcessTransport {
    fn kind(&self) -> TransportKind {
        TransportKind::InProcess
    }

    fn team_id(&self) -> &str {
        self.bus.team_id()
    }

    async fn start(&self) -> Result<()> {
        self.bus.start().await
    }

    async fn stop(&self) -> Result<()> {
        self.bus.stop().await
    }

    async fn send(&self, envelope: MessageEnvelope, timeout: Option<Duration>) -> Result<Value> {
        // The caller's deadline is passed through unchanged: the bus applies its
        // own `process_timeout` only when the transport call has none.
        self.bus.send(envelope, timeout).await
    }

    async fn publish(&self, envelope: MessageEnvelope) -> Result<()> {
        self.bus.publish(envelope).await
    }

    fn add_subscription(&self, agent_id: &str, topic: &str) -> Result<()> {
        // The in-process registry cannot fail, so the pair is always observed by
        // the bus that also fans `publish` out.
        self.bus.add_subscription(agent_id, topic);
        Ok(())
    }

    fn remove_subscription(&self, agent_id: &str, topic: &str) -> Result<()> {
        self.bus.remove_subscription(agent_id, topic);
        Ok(())
    }

    fn subscribers(&self, topic_id: &str) -> Vec<String> {
        self.bus.subscribers(topic_id)
    }

    fn list_subscriptions(&self) -> BTreeMap<String, Vec<String>> {
        self.bus.list_subscriptions()
    }

    fn agent_subscriptions(&self, agent_id: &str) -> Vec<String> {
        self.bus.list_agent_subscriptions(agent_id)
    }

    fn subscription_count(&self) -> usize {
        self.bus.get_subscription_count()
    }
}
