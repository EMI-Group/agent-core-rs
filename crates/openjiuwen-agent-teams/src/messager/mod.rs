//! Transport seam for team events: topic publish/subscribe plus point-to-point delivery.
//!
//! Python parity source: `openjiuwen/agent_teams/messager/` (`messager.py` ABC, `base.py` config +
//! factory, `inprocess.py` handler bus). Every event that crosses a member boundary goes through a
//! [`Messager`]; no other module of this crate opens a socket, queue or channel for team messaging.
//!
//! Only the in-process transport is implemented ([`InProcessMessager`], built by
//! [`create_messager`] for `TransportSpec { kind: "inprocess" }`). The `pyzmq` and `hybrid` backends
//! stay documented seams that reject construction — see the crate `CONTEXT.md` "Deferred" list.
//!
//! Module map:
//!
//! | Module | Responsibility |
//! | --- | --- |
//! | [`inprocess`] | [`InProcessMessager`]: the per-instance in-process bus |
//! | [`factory`] | transport config carriers and [`create_messager`] |
//! | `distributed` | feature-gated `PyZmqMessager` stub (deferred capability) |
//!
//! The registry is per-instance, not process-global: two messagers never see each other's
//! subscriptions, and the runtime shares one `Arc<dyn Messager>` per team.

use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::Result;

use crate::schema::{EventMessage, TeamTopic};

#[cfg(feature = "distributed")]
pub mod distributed;
pub mod factory;
pub mod inprocess;

#[cfg(feature = "distributed")]
pub use distributed::PyZmqMessager;
pub use factory::{
    create_messager, InProcessMessagerConfig, MessagerPeerConfig, MessagerTransportConfig,
    PyZmqMessagerConfig,
};
pub use inprocess::InProcessMessager;

/// Synchronous handler invoked for one delivered [`EventMessage`].
///
/// The sink receives a copy of the envelope and must treat it as read-only: one publish reaches
/// several subscribers, and each of them gets its own clone.
pub type EventSink = Arc<dyn Fn(EventMessage) + Send + Sync>;

/// Pluggable transport used to move team events between members.
///
/// Implementations own their registry; every method is cheap and non-blocking for the in-process
/// transport. Errors are only raised for caller mistakes (unknown target is deliberately *not* an
/// error — reachability is decided by mailbox persistence, not by the transport).
#[async_trait]
pub trait Messager: Send + Sync {
    /// Returns the Python transport kind name (`"inprocess"`).
    fn name(&self) -> &'static str;

    /// Subscribes `member` to every topic in `topics`, using `sink` for all of them.
    ///
    /// Idempotent per `(member, topic)`: re-subscribing replaces that member's sink. A non-empty
    /// `topics` also installs `sink` as the member's direct (point-to-point) handler.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error`] when the transport rejects the subscription.
    fn subscribe(&self, member: &str, topics: &[TeamTopic], sink: EventSink) -> Result<()>;

    /// Removes `member` from every topic and from the direct-delivery registry.
    ///
    /// Unknown members are a no-op returning `Ok(())`.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error`] when the transport rejects the removal.
    fn unsubscribe(&self, member: &str) -> Result<()>;

    /// Publishes `event` to every subscriber of `topic`, in sorted member order.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error`] when the transport cannot publish. A panicking
    /// sink is logged and does not fail the publish.
    async fn publish(&self, topic: TeamTopic, event: EventMessage) -> Result<()>;

    /// Delivers `event` to exactly one member's direct handler.
    ///
    /// An unknown target is a no-op returning `Ok(())`.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error`] when the transport cannot deliver. A panicking
    /// handler is logged and does not fail the send.
    async fn send(&self, target: &str, event: EventMessage) -> Result<()>;

    /// Records that this messager serves `team_name` (the Python leader hook).
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error`] when the transport rejects the session join.
    async fn subscribe_transport(&self, team_name: &str) -> Result<()>;

    /// Returns the members subscribed to `topic`, sorted by member name.
    fn subscribers(&self, topic: TeamTopic) -> Vec<String>;

    /// Returns `true` while the transport is delivering.
    fn is_started(&self) -> bool;

    /// Starts delivery; idempotent.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error`] when the transport cannot start.
    async fn start(&self) -> Result<()>;

    /// Stops delivery without dropping subscriptions; idempotent.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error`] when the transport cannot stop.
    async fn stop(&self) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trait_is_object_safe_and_names_the_python_kind() {
        let messager: Arc<dyn Messager> = Arc::new(InProcessMessager::new());
        assert_eq!(messager.name(), "inprocess");
        assert!(messager.is_started());
    }
}
