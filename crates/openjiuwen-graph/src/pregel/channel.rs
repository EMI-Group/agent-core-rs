//! CNF barrier channels and the superstep message buffer.
//!
//! Every node owns one channel. Its *groups* are a conjunction: the node is
//! ready when every group contains at least one source that has fired. A node
//! with no groups is a trigger channel: any fired source makes it ready.
//!
//! Activation is *accept-then-activate*: routing messages produced by a
//! superstep are queued in [`ChannelManager`]'s buffer and only become visible
//! when the executor flushes them after every node of the superstep completed.

use std::collections::{BTreeMap, BTreeSet};

use openjiuwen_common::error::{Error, Result};

use crate::store::{ChannelMessage, ChannelSnapshot};
use crate::START_NODE;

/// One node's barrier channel.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct NodeChannel {
    groups: Vec<BTreeSet<String>>,
    fired: Vec<BTreeSet<String>>,
    triggers: BTreeSet<String>,
}

impl NodeChannel {
    /// Builds a channel with the compiled CNF `groups`.
    pub(crate) fn new(groups: Vec<BTreeSet<String>>) -> Self {
        let fired = vec![BTreeSet::new(); groups.len()];
        Self {
            groups,
            fired,
            triggers: BTreeSet::new(),
        }
    }

    /// Reports whether a trigger fired for this channel.
    #[cfg(test)]
    pub(crate) fn triggers(&self) -> &BTreeSet<String> {
        &self.triggers
    }

    /// Records that `source` activated this channel.
    ///
    /// Entry seeding ([`START_NODE`]) is an unconditional activation, so an
    /// explicitly declared entry that also has incoming edges still runs in the
    /// first superstep. A channel without groups accepts any source.
    pub(crate) fn accept(&mut self, source: &str) {
        if self.groups.is_empty() || source == START_NODE {
            self.triggers.insert(source.to_string());
            return;
        }
        for (index, group) in self.groups.iter().enumerate() {
            if group.contains(source) {
                if let Some(fired) = self.fired.get_mut(index) {
                    fired.insert(source.to_string());
                }
            }
        }
    }

    /// Consumes a ready channel, clearing every fired source.
    pub(crate) fn consume(&mut self) {
        for fired in &mut self.fired {
            fired.clear();
        }
        self.triggers.clear();
    }

    /// Reports whether the channel's CNF condition is satisfied.
    pub(crate) fn ready(&self) -> bool {
        if !self.triggers.is_empty() {
            return true;
        }
        if self.groups.is_empty() {
            return false;
        }
        self.groups.iter().enumerate().all(|(index, group)| {
            !group.is_empty() && self.fired.get(index).is_some_and(|fired| !fired.is_empty())
        })
    }

    /// Captures the channel state for a snapshot.
    pub(crate) fn snapshot(&self) -> ChannelSnapshot {
        ChannelSnapshot {
            groups: self
                .groups
                .iter()
                .map(|group| group.iter().cloned().collect())
                .collect(),
            fired: self
                .fired
                .iter()
                .map(|fired| fired.iter().cloned().collect())
                .collect(),
            triggers: self.triggers.iter().cloned().collect(),
        }
    }

    /// Restores a snapshot into this (compiled) channel.
    ///
    /// # Errors
    ///
    /// Returns [`Error::graph`] when the snapshot's channel topology does not
    /// match the compiled graph.
    pub(crate) fn restore(&mut self, snapshot: &ChannelSnapshot) -> Result<()> {
        let groups: Vec<BTreeSet<String>> = snapshot
            .groups
            .iter()
            .map(|group| group.iter().cloned().collect())
            .collect();
        if groups != self.groups {
            return Err(Error::graph(
                "graph snapshot channel topology does not match the compiled graph",
            ));
        }
        if snapshot.fired.len() != self.groups.len() {
            return Err(Error::graph(
                "graph snapshot channel state does not match the compiled graph",
            ));
        }
        self.fired = snapshot
            .fired
            .iter()
            .map(|fired| fired.iter().cloned().collect())
            .collect();
        self.triggers = snapshot.triggers.iter().cloned().collect();
        Ok(())
    }
}

/// Every node's channel plus the accept-then-activate buffer.
#[derive(Debug, Clone, Default)]
pub(crate) struct ChannelManager {
    channels: BTreeMap<String, NodeChannel>,
    buffer: Vec<ChannelMessage>,
}

impl ChannelManager {
    /// Builds a manager from the compiled channels.
    pub(crate) fn new(channels: BTreeMap<String, NodeChannel>) -> Self {
        Self {
            channels,
            buffer: Vec::new(),
        }
    }

    /// Queues a routing message for `target`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::graph`] when `target` is not a node of the graph.
    pub(crate) fn accept(&mut self, target: &str, source: &str) -> Result<()> {
        if !self.channels.contains_key(target) {
            return Err(Error::graph(format!(
                "graph routed to unknown node `{target}`"
            )));
        }
        self.buffer.push(ChannelMessage {
            target: target.to_string(),
            source: source.to_string(),
        });
        Ok(())
    }

    /// Applies every buffered message to its target channel.
    ///
    /// # Errors
    ///
    /// Returns [`Error::graph`] when a buffered message addresses an unknown
    /// node (only possible with a hand-crafted snapshot buffer).
    pub(crate) fn flush(&mut self) -> Result<()> {
        let messages = std::mem::take(&mut self.buffer);
        for message in messages {
            let channel = self.channels.get_mut(&message.target).ok_or_else(|| {
                Error::graph(format!("graph routed to unknown node `{}`", message.target))
            })?;
            channel.accept(&message.source);
        }
        Ok(())
    }

    /// Returns the ready node ids in declaration order.
    pub(crate) fn ready_nodes(&self, order: &[String]) -> Vec<String> {
        order
            .iter()
            .filter(|id| self.channels.get(*id).is_some_and(NodeChannel::ready))
            .cloned()
            .collect()
    }

    /// Consumes `id`'s activation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::graph`] when `id` is not a node of the graph.
    pub(crate) fn consume(&mut self, id: &str) -> Result<()> {
        let channel = self
            .channels
            .get_mut(id)
            .ok_or_else(|| Error::graph(format!("unknown node `{id}`")))?;
        channel.consume();
        Ok(())
    }

    /// Captures every channel's state.
    pub(crate) fn snapshot(&self) -> BTreeMap<String, ChannelSnapshot> {
        self.channels
            .iter()
            .map(|(id, channel)| (id.clone(), channel.snapshot()))
            .collect()
    }

    /// Restores every channel's state.
    ///
    /// # Errors
    ///
    /// Returns [`Error::graph`] when a channel is missing from the snapshot or
    /// its topology does not match.
    pub(crate) fn restore(&mut self, snapshot: &BTreeMap<String, ChannelSnapshot>) -> Result<()> {
        for (id, channel) in &mut self.channels {
            let saved = snapshot
                .get(id)
                .ok_or_else(|| Error::graph(format!("graph snapshot is missing channel `{id}`")))?;
            channel.restore(saved)?;
        }
        Ok(())
    }

    /// Returns the pending (unflushed) messages.
    pub(crate) fn buffer_snapshot(&self) -> Vec<ChannelMessage> {
        self.buffer.clone()
    }

    /// Restores the pending (unflushed) messages.
    pub(crate) fn restore_buffer(&mut self, buffer: Vec<ChannelMessage>) {
        self.buffer = buffer;
    }

    /// Returns the number of buffered messages.
    #[cfg(test)]
    pub(crate) fn buffered(&self) -> usize {
        self.buffer.len()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;

    fn channel(groups: Vec<Vec<&str>>) -> NodeChannel {
        NodeChannel::new(
            groups
                .into_iter()
                .map(|group| group.into_iter().map(str::to_string).collect())
                .collect(),
        )
    }

    fn build() -> ChannelManager {
        let mut channels = BTreeMap::new();
        channels.insert("a".to_string(), NodeChannel::new(Vec::new()));
        channels.insert("b".to_string(), channel(vec![vec!["a"]]));
        channels.insert("c".to_string(), channel(vec![vec!["a"], vec!["b"]]));
        ChannelManager::new(channels)
    }

    #[test]
    fn trigger_channel_is_ready_once_any_source_fires() {
        let mut trigger = NodeChannel::new(Vec::new());
        assert!(!trigger.ready());
        trigger.accept("__start__");
        assert!(trigger.ready());
        assert_eq!(trigger.triggers().len(), 1);
        trigger.consume();
        assert!(!trigger.ready());
    }

    #[test]
    fn single_group_channel_waits_for_its_source() {
        let mut node = channel(vec![vec!["a"]]);
        assert!(!node.ready());
        node.accept("other");
        assert!(!node.ready());
        node.accept("a");
        assert!(node.ready());
        node.consume();
        assert!(!node.ready());
    }

    #[test]
    fn entry_seeding_activates_a_channel_with_incoming_edges() {
        let mut node = channel(vec![vec!["a"]]);
        assert!(!node.ready());
        node.accept(crate::START_NODE);
        assert!(node.ready());
        node.consume();
        assert!(!node.ready());
        node.accept("a");
        assert!(node.ready());
    }

    #[test]
    fn barrier_channel_requires_every_group() {
        let mut node = channel(vec![vec!["a"], vec!["b"]]);
        node.accept("a");
        assert!(!node.ready());
        node.accept("b");
        assert!(node.ready());
        node.consume();
        assert!(!node.ready());
    }

    #[test]
    fn empty_group_never_becomes_ready() {
        let node = channel(vec![vec![]]);
        assert!(!node.ready());
    }

    #[test]
    fn buffer_applies_only_on_flush() {
        let mut manager = build();
        assert!(manager
            .ready_nodes(&["a".to_string(), "b".to_string()])
            .is_empty());
        manager.accept("b", "a").unwrap();
        assert_eq!(manager.buffered(), 1);
        assert!(manager.ready_nodes(&["b".to_string()]).is_empty());
        manager.flush().unwrap();
        assert_eq!(manager.buffered(), 0);
        assert_eq!(
            manager.ready_nodes(&["a".to_string(), "b".to_string()]),
            vec!["b".to_string()]
        );
    }

    #[test]
    fn accept_rejects_unknown_targets() {
        let mut manager = build();
        let error = manager.accept("zzz", "a").unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Graph);
    }

    #[test]
    fn ready_nodes_follow_declaration_order() {
        let mut manager = build();
        manager.accept("b", "a").unwrap();
        manager.accept("a", "__start__").unwrap();
        manager.flush().unwrap();
        assert_eq!(
            manager.ready_nodes(&["a".to_string(), "b".to_string()]),
            vec!["a".to_string(), "b".to_string()]
        );
        assert!(manager.ready_nodes(&["c".to_string()]).is_empty());
    }

    #[test]
    fn snapshot_round_trips() {
        let mut manager = build();
        manager.accept("b", "a").unwrap();
        manager.flush().unwrap();
        let snapshot = manager.snapshot();
        assert_eq!(snapshot["b"].fired, vec![vec!["a".to_string()]]);

        let mut restored = build();
        restored.consume("b").unwrap();
        assert!(restored.ready_nodes(&["b".to_string()]).is_empty());
        restored.restore(&snapshot).unwrap();
        assert_eq!(
            restored.ready_nodes(&["b".to_string()]),
            vec!["b".to_string()]
        );
    }

    #[test]
    fn restore_rejects_topology_mismatch() {
        let snapshot = NodeChannel::new(Vec::new()).snapshot();
        let mut node = channel(vec![vec!["a"]]);
        let error = node.restore(&snapshot).unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Graph);

        let mut manager = build();
        let mut incomplete = manager.snapshot();
        incomplete.remove("b");
        assert!(manager.restore(&incomplete).is_err());
    }

    #[test]
    fn buffer_snapshot_and_restore() {
        let mut manager = build();
        manager.accept("c", "a").unwrap();
        let buffer = manager.buffer_snapshot();
        let mut restored = build();
        restored.restore_buffer(buffer);
        restored.flush().unwrap();
        assert_eq!(
            restored.ready_nodes(&["c".to_string()]),
            Vec::<String>::new()
        );
    }
}
