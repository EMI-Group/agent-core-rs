//! Group addressing, broadcast recipients, and a serializable routing view.
//!
//! This module is an additive Rust extension, not a port: the Python
//! `core.multi_agent` package has no group, broadcast, or inbox layer (its
//! `MessageRouter` only classifies messages by transport-topic suffix). The
//! contract here is inspired by the `agent_teams` product, where a broadcast is
//! addressed with a sentinel and the sender is always excluded from the
//! recipient set.
//!
//! Everything in this module is process-local and synchronous: the
//! [`GroupRegistry`] holds a short-lived `parking_lot` lock, and the
//! [`RoutingTable`] is a plain serializable snapshot a caller can log or ship
//! over a wire.

use std::collections::{BTreeMap, BTreeSet};

use openjiuwen_common::error::{Error, Result};
use parking_lot::RwLock;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::subscription::topic_matches;

/// Metadata key stamped on every envelope delivered by a group broadcast.
pub const GROUP_METADATA_KEY: &str = "group";

/// Named groups of agent ids.
///
/// A group is a routing alias: [`TeamRuntime::broadcast_to_group`] resolves it to
/// its members, excludes the sender, and dispatches to the members still
/// registered with the runtime. Group ids and member ids are stored verbatim;
/// [`GroupRegistry::group_members`] and [`GroupRegistry::groups`] return sorted
/// lists, so routing stays deterministic.
///
/// [`TeamRuntime::broadcast_to_group`]: crate::TeamRuntime::broadcast_to_group
#[derive(Debug, Default)]
pub struct GroupRegistry {
    groups: RwLock<BTreeMap<String, BTreeSet<String>>>,
}

impl GroupRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers (or replaces) `group_id` with `members` and returns the
    /// resulting member count.
    ///
    /// Members are deduplicated and kept sorted. Replacing an existing group
    /// discards its old membership entirely.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when `group_id` is empty.
    pub fn register_group(
        &self,
        group_id: impl Into<String>,
        members: impl IntoIterator<Item = String>,
    ) -> Result<usize> {
        let group_id = group_id.into();
        if group_id.is_empty() {
            return Err(Error::validation("group_id is required"));
        }
        let members: BTreeSet<String> = members.into_iter().collect();
        let count = members.len();
        self.groups.write().insert(group_id, members);
        Ok(count)
    }

    /// Adds `agent_id` to `group_id`, creating an empty group implicitly.
    ///
    /// Returns `true` when the membership is new and `false` when the agent was
    /// already a member.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when `group_id` or `agent_id` is empty.
    pub fn add_member(&self, group_id: &str, agent_id: &str) -> Result<bool> {
        if group_id.is_empty() {
            return Err(Error::validation("group_id is required"));
        }
        if agent_id.is_empty() {
            return Err(Error::validation("agent_id is required"));
        }
        Ok(self
            .groups
            .write()
            .entry(group_id.to_owned())
            .or_default()
            .insert(agent_id.to_owned()))
    }

    /// Removes `agent_id` from `group_id`, returning `true` when it was a
    /// member.
    ///
    /// The (possibly empty) group entry is kept until
    /// [`unregister_group`](Self::unregister_group) removes it.
    pub fn remove_member(&self, group_id: &str, agent_id: &str) -> bool {
        self.groups
            .write()
            .get_mut(group_id)
            .is_some_and(|members| members.remove(agent_id))
    }

    /// Removes the whole group, returning `true` when it existed.
    pub fn unregister_group(&self, group_id: &str) -> bool {
        self.groups.write().remove(group_id).is_some()
    }

    /// Returns the sorted members of `group_id` (empty for an unknown group).
    #[must_use]
    pub fn group_members(&self, group_id: &str) -> Vec<String> {
        self.groups
            .read()
            .get(group_id)
            .map(|members| members.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Returns every group id, sorted.
    #[must_use]
    pub fn groups(&self) -> Vec<String> {
        self.groups.read().keys().cloned().collect()
    }

    /// Returns the sorted ids of the groups `agent_id` belongs to.
    #[must_use]
    pub fn groups_of(&self, agent_id: &str) -> Vec<String> {
        self.groups
            .read()
            .iter()
            .filter(|(_, members)| members.contains(agent_id))
            .map(|(group, _)| group.clone())
            .collect()
    }

    /// Returns the number of registered groups.
    #[must_use]
    pub fn group_count(&self) -> usize {
        self.groups.read().len()
    }

    /// Returns the total number of `(group, member)` pairs.
    #[must_use]
    pub fn member_count(&self) -> usize {
        self.groups.read().values().map(BTreeSet::len).sum()
    }

    /// Removes every group, returning how many were removed.
    pub fn clear(&self) -> usize {
        let mut groups = self.groups.write();
        let count = groups.len();
        groups.clear();
        count
    }
}

/// Computes the recipient set of a broadcast.
///
/// The result is sorted and deduplicated; `sender` is excluded unless
/// `include_sender` is set. When `sender` is `None` nothing is excluded.
#[must_use]
pub fn broadcast_targets(
    members: impl IntoIterator<Item = String>,
    sender: Option<&str>,
    include_sender: bool,
) -> Vec<String> {
    let mut targets: BTreeSet<String> = members.into_iter().collect();
    if !include_sender {
        if let Some(sender) = sender {
            targets.remove(sender);
        }
    }
    targets.into_iter().collect()
}

/// One delivery of a group broadcast.
///
/// Process-local: the outcome carries the member's reply or its error and is
/// therefore not serializable.
#[derive(Debug)]
pub struct GroupDelivery {
    /// Id of the member the message was dispatched to.
    pub agent_id: String,
    /// The member's reply value, or the error it failed with.
    pub outcome: Result<Value>,
}

/// The outcome of one [`TeamRuntime::broadcast_to_group`] call.
///
/// A member failure never aborts the batch: every recipient still gets a
/// [`GroupDelivery`], and [`failed`](Self::failed) counts the errors.
///
/// [`TeamRuntime::broadcast_to_group`]: crate::TeamRuntime::broadcast_to_group
#[derive(Debug)]
pub struct GroupBroadcast {
    /// The group the broadcast was addressed to.
    pub group: String,
    /// Sorted ids of the members that were dispatched to.
    pub recipients: Vec<String>,
    /// Sorted ids of group members that were skipped because they are not
    /// registered with the runtime.
    pub skipped: Vec<String>,
    /// One entry per recipient, in recipient order.
    pub deliveries: Vec<GroupDelivery>,
}

impl GroupBroadcast {
    /// Returns how many deliveries succeeded.
    #[must_use]
    pub fn succeeded(&self) -> usize {
        self.deliveries
            .iter()
            .filter(|delivery| delivery.outcome.is_ok())
            .count()
    }

    /// Returns how many deliveries failed.
    #[must_use]
    pub fn failed(&self) -> usize {
        self.deliveries
            .iter()
            .filter(|delivery| delivery.outcome.is_err())
            .count()
    }

    /// Returns the reply of `agent_id` when its delivery succeeded.
    #[must_use]
    pub fn value_of(&self, agent_id: &str) -> Option<&Value> {
        self.deliveries
            .iter()
            .find(|delivery| delivery.agent_id == agent_id)
            .and_then(|delivery| delivery.outcome.as_ref().ok())
    }

    /// Consumes the broadcast into one `agent id -> outcome` map.
    #[must_use]
    pub fn into_results(self) -> BTreeMap<String, Result<Value>> {
        self.deliveries
            .into_iter()
            .map(|delivery| (delivery.agent_id, delivery.outcome))
            .collect()
    }
}

/// A serializable snapshot of one runtime's routing state.
///
/// `groups` maps a group id to its sorted members, `topics` maps a topic
/// pattern to its sorted subscribers, and `agents` maps an agent id to its
/// sorted destinations (the patterns it subscribes to plus the groups it
/// belongs to).
///
/// The snapshot is plain data: assemble it with
/// [`TeamRuntime::routing_table`](crate::TeamRuntime::routing_table) and ship or
/// log it freely.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct RoutingTable {
    /// Team id the snapshot was taken from.
    pub team_id: String,
    /// Group id -> sorted members.
    pub groups: BTreeMap<String, Vec<String>>,
    /// Topic pattern -> sorted subscriber ids.
    pub topics: BTreeMap<String, Vec<String>>,
    /// Agent id -> sorted destinations (topics plus groups).
    pub agents: BTreeMap<String, Vec<String>>,
}

impl RoutingTable {
    /// Resolves `destination` to the ids it addresses, sorted and deduplicated.
    ///
    /// A known group id resolves to its members. Otherwise `destination` is
    /// matched against every topic pattern with
    /// `crate::subscription::topic_matches`, so a concrete
    /// topic such as `news.tech` also resolves through a `news.*` subscription.
    /// An unknown destination resolves to an empty vector.
    #[must_use]
    pub fn route(&self, destination: &str) -> Vec<String> {
        if let Some(members) = self.groups.get(destination) {
            let unique: BTreeSet<String> = members.iter().cloned().collect();
            return unique.into_iter().collect();
        }
        let mut resolved = BTreeSet::new();
        for (pattern, subscribers) in &self.topics {
            if topic_matches(pattern, destination) {
                resolved.extend(subscribers.iter().cloned());
            }
        }
        resolved.into_iter().collect()
    }

    /// Returns the sorted destinations (topics plus groups) of `agent_id`.
    #[must_use]
    pub fn destinations(&self, agent_id: &str) -> Vec<String> {
        self.agents.get(agent_id).cloned().unwrap_or_default()
    }
}

/// Builds a routing snapshot from the parts a runtime exposes.
///
/// Kept crate-internal: callers use the runtime's
/// [`routing_table`](crate::TeamRuntime::routing_table) method.
pub(crate) fn assemble_routing_table(
    team_id: &str,
    groups: &GroupRegistry,
    topics: BTreeMap<String, Vec<String>>,
) -> RoutingTable {
    let group_snapshot: BTreeMap<String, Vec<String>> = groups
        .groups()
        .into_iter()
        .map(|group| {
            let members = groups.group_members(&group);
            (group, members)
        })
        .collect();
    let mut agents: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (pattern, subscribers) in &topics {
        for agent in subscribers {
            agents
                .entry(agent.clone())
                .or_default()
                .insert(pattern.clone());
        }
    }
    for (group, members) in &group_snapshot {
        for agent in members {
            agents
                .entry(agent.clone())
                .or_default()
                .insert(group.clone());
        }
    }
    let agents = agents
        .into_iter()
        .map(|(agent, destinations)| (agent, destinations.into_iter().collect()))
        .collect();
    RoutingTable {
        team_id: team_id.to_owned(),
        groups: group_snapshot,
        topics,
        agents,
    }
}
