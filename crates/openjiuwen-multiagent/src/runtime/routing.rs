//! Group routing, routing-table snapshots, and inbox-filter wiring.
//!
//! Kept apart from the registry and dispatch core so [`TeamRuntime`] stays a
//! small surface. The group registry and the inbox-filter registry are
//! process-local handles created with the runtime; every method takes a short
//! lock scope, and [`TeamRuntime::broadcast_to_group`] never holds one across an
//! `.await`.
//!
//! This is an additive Rust extension: the Python `core.multi_agent` package has
//! no group or inbox layer. The broadcast contract (a named recipient set with
//! the sender excluded) is inspired by the `agent_teams` product.

use std::sync::Arc;
use std::time::Duration;

use futures::future::join_all;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::id::uuid_v4;
use serde_json::Value;

use super::TeamRuntime;
use crate::envelope::MessageEnvelope;
use crate::events::TeamEvent;
use crate::inbox::{InboxFilter, InboxFilterRegistry};
use crate::routing::{
    assemble_routing_table, broadcast_targets, GroupBroadcast, GroupDelivery, GroupRegistry,
    RoutingTable, GROUP_METADATA_KEY,
};

impl TeamRuntime {
    /// Returns the runtime's group registry.
    #[must_use]
    pub fn groups(&self) -> Arc<GroupRegistry> {
        Arc::clone(&self.groups)
    }

    /// Returns the runtime's inbox-filter registry.
    #[must_use]
    pub fn inbox_filters(&self) -> Arc<InboxFilterRegistry> {
        Arc::clone(&self.inbox_filters)
    }

    /// Registers (or replaces) a named group and returns its member count.
    ///
    /// See [`GroupRegistry::register_group`].
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when `group_id` is empty.
    pub fn register_group(
        &self,
        group_id: impl Into<String>,
        members: impl IntoIterator<Item = String>,
    ) -> Result<usize> {
        self.groups.register_group(group_id, members)
    }

    /// Removes a named group, returning `true` when it existed.
    pub fn unregister_group(&self, group_id: &str) -> bool {
        self.groups.unregister_group(group_id)
    }

    /// Returns the sorted members of `group_id`.
    #[must_use]
    pub fn group_members(&self, group_id: &str) -> Vec<String> {
        self.groups.group_members(group_id)
    }

    /// Returns the sorted groups `agent_id` belongs to.
    #[must_use]
    pub fn groups_of(&self, agent_id: &str) -> Vec<String> {
        self.groups.groups_of(agent_id)
    }

    /// Installs the inbox filter of `agent_id`, replacing an existing one.
    pub fn set_inbox_filter(&self, agent_id: impl Into<String>, filter: InboxFilter) {
        self.inbox_filters.set(agent_id, filter);
    }

    /// Removes the inbox filter of `agent_id`, returning `true` when one
    /// existed.
    pub fn remove_inbox_filter(&self, agent_id: &str) -> bool {
        self.inbox_filters.remove(agent_id)
    }

    /// Returns a copy of the inbox filter of `agent_id`.
    #[must_use]
    pub fn inbox_filter(&self, agent_id: &str) -> Option<InboxFilter> {
        self.inbox_filters.get(agent_id)
    }

    /// Assembles a serializable snapshot of the runtime's groups, topic
    /// subscriptions, and per-agent destinations.
    #[must_use]
    pub fn routing_table(&self) -> RoutingTable {
        assemble_routing_table(self.team_id(), &self.groups, self.list_subscriptions())
    }

    /// Broadcasts `message` to every registered member of `group_id`.
    ///
    /// Validation mirrors [`TeamRuntime::send`]: an empty sender is rejected.
    /// The recipient set is [`broadcast_targets`] with the sender excluded,
    /// intersected with the agents registered with this runtime; a group member
    /// that is not registered is skipped (logged at `warn`) and reported in
    /// [`GroupBroadcast::skipped`].
    ///
    /// One [`TeamEvent::message_sent`] is emitted before dispatch, with
    /// `detail.kind = "group"`, `detail.group`, and `detail.recipients`. Each
    /// delivery carries `metadata["group"] = group_id` and runs concurrently
    /// over the runtime's transport; a member failure is recorded in its
    /// [`GroupDelivery`] and never aborts the batch.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an empty sender, [`Error::NotFound`] for an
    /// unknown group. Delivery failures are returned inside the broadcast, not
    /// as an error.
    pub async fn broadcast_to_group(
        &self,
        message: Value,
        group_id: &str,
        sender: &str,
        session_id: Option<&str>,
        timeout: Option<Duration>,
    ) -> Result<GroupBroadcast> {
        if sender.is_empty() {
            return Err(Error::validation("sender is required for message tracing"));
        }
        if !self.groups.groups().iter().any(|group| group == group_id) {
            return Err(Error::not_found(format!(
                "group '{group_id}' is not registered"
            )));
        }
        let members = self.groups.group_members(group_id);
        let targets = broadcast_targets(members, Some(sender), false);
        let mut recipients = Vec::new();
        let mut skipped = Vec::new();
        for agent_id in targets {
            if self.has_agent(&agent_id) {
                recipients.push(agent_id);
            } else {
                tracing::warn!(
                    group = group_id,
                    agent_id,
                    "skipping group member that is not registered with the runtime"
                );
                skipped.push(agent_id);
            }
        }
        let event = TeamEvent::message_sent(self.team_id(), sender, group_id, uuid_v4())
            .with_detail("kind", Value::String("group".to_owned()))
            .with_detail("group", Value::String(group_id.to_owned()))
            .with_detail("recipients", Value::from(recipients.len()));
        self.events.emit(event);
        let deadline = Some(timeout.unwrap_or_else(|| self.p2p_timeout()));
        let transport = self.transport();
        let dispatches = recipients.iter().map(|recipient| {
            let mut envelope = MessageEnvelope::p2p(message.clone(), sender, recipient.clone())
                .with_metadata(GROUP_METADATA_KEY, Value::String(group_id.to_owned()));
            if let Some(session_id) = session_id {
                envelope = envelope.with_session(session_id);
            }
            let transport = Arc::clone(&transport);
            let agent_id = recipient.clone();
            async move {
                let outcome = transport.send(envelope, deadline).await;
                GroupDelivery { agent_id, outcome }
            }
        });
        let deliveries = join_all(dispatches).await;
        Ok(GroupBroadcast {
            group: group_id.to_owned(),
            recipients,
            skipped,
            deliveries,
        })
    }
}
