//! Per-agent inbox filters consulted before a delivery reaches its member.
//!
//! This module is an additive Rust extension, not a port: the Python
//! `core.multi_agent` package has no inbox-filter layer (every bus delivery
//! reaches its registered member). The matching rules here are inspired by the
//! `agent_teams` broadcast contract, where a receiver is only eligible when the
//! sender and channel are acceptable.
//!
//! An [`InboxFilter`] is pure data, so it serializes to JSON; a
//! [`InboxFilterRegistry`] maps agent ids to filters and is process-local.

use std::collections::{BTreeMap, BTreeSet};

use parking_lot::RwLock;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::envelope::MessageEnvelope;
use crate::subscription::topic_matches;

/// One agent's acceptance rules for inbound deliveries.
///
/// The default accepts everything. Rules combine as follows, in order:
///
/// 1. a sender in [`deny_senders`](Self::deny_senders) is rejected (deny wins);
/// 2. when [`allow_senders`](Self::allow_senders) is set, the sender must be in
///    it (an envelope without a sender is rejected);
/// 3. when [`topics`](Self::topics) is non-empty, the envelope must carry a
///    topic id matching one of the patterns;
/// 4. every [`metadata`](Self::metadata) pair must be present with the exact
///    string value;
/// 5. [`require_session`](Self::require_session) requires a session id.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct InboxFilter {
    /// Accepted sender ids; `None` accepts every sender.
    pub allow_senders: Option<BTreeSet<String>>,
    /// Rejected sender ids; checked before the allow-list.
    pub deny_senders: BTreeSet<String>,
    /// Accepted topic patterns, matched with
    /// `crate::subscription::topic_matches`.
    pub topics: Vec<String>,
    /// Metadata entries the envelope must carry exactly.
    pub metadata: BTreeMap<String, String>,
    /// Whether the envelope must carry a session id.
    pub require_session: bool,
}

impl InboxFilter {
    /// Creates an all-accepting filter.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Restricts inbound deliveries to `senders`.
    ///
    /// Repeated calls accumulate. An envelope without a sender is rejected once
    /// an allow-list is set.
    #[must_use]
    pub fn allow_senders<I, S>(mut self, senders: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.allow_senders
            .get_or_insert_default()
            .extend(senders.into_iter().map(Into::into));
        self
    }

    /// Rejects inbound deliveries from `senders`.
    ///
    /// Repeated calls accumulate; the deny-list is checked before the
    /// allow-list.
    #[must_use]
    pub fn deny_senders<I, S>(mut self, senders: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.deny_senders
            .extend(senders.into_iter().map(Into::into));
        self
    }

    /// Restricts inbound deliveries to envelopes whose topic matches one of
    /// `patterns`.
    ///
    /// Repeated calls accumulate; a point-to-point envelope without a topic id
    /// is rejected once a pattern is set.
    #[must_use]
    pub fn allow_topics<I, S>(mut self, patterns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.topics.extend(patterns.into_iter().map(Into::into));
        self
    }

    /// Requires the envelope to carry `key` with the exact string `value`.
    #[must_use]
    pub fn require_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Requires the envelope to carry a session id.
    #[must_use]
    pub fn require_session(mut self) -> Self {
        self.require_session = true;
        self
    }

    /// Returns `true` when `envelope` satisfies every rule of this filter.
    #[must_use]
    pub fn accepts(&self, envelope: &MessageEnvelope) -> bool {
        let sender = envelope.sender.as_deref();
        if sender.is_some_and(|sender| self.deny_senders.contains(sender)) {
            return false;
        }
        if let Some(allowed) = &self.allow_senders {
            if !sender.is_some_and(|sender| allowed.contains(sender)) {
                return false;
            }
        }
        if !self.topics.is_empty() {
            let Some(topic) = envelope.topic_id.as_deref() else {
                return false;
            };
            if !self
                .topics
                .iter()
                .any(|pattern| topic_matches(pattern, topic))
            {
                return false;
            }
        }
        for (key, expected) in &self.metadata {
            let Some(actual) = envelope.metadata.get(key) else {
                return false;
            };
            if actual.as_str() != Some(expected.as_str()) {
                return false;
            }
        }
        if self.require_session && envelope.session_id.is_none() {
            return false;
        }
        true
    }
}

/// The inbox filters of one runtime, keyed by recipient agent id.
///
/// An agent without an entry accepts every delivery, so a runtime with an empty
/// registry behaves exactly like one without this layer.
#[derive(Debug, Default)]
pub struct InboxFilterRegistry {
    filters: RwLock<BTreeMap<String, InboxFilter>>,
}

impl InboxFilterRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Installs `filter` for `agent_id`, replacing an existing one.
    pub fn set(&self, agent_id: impl Into<String>, filter: InboxFilter) {
        self.filters.write().insert(agent_id.into(), filter);
    }

    /// Removes the filter of `agent_id`, returning `true` when one existed.
    pub fn remove(&self, agent_id: &str) -> bool {
        self.filters.write().remove(agent_id).is_some()
    }

    /// Returns a copy of the filter of `agent_id`.
    #[must_use]
    pub fn get(&self, agent_id: &str) -> Option<InboxFilter> {
        self.filters.read().get(agent_id).cloned()
    }

    /// Returns `true` when `agent_id` accepts `envelope`; an agent without a
    /// filter accepts everything.
    #[must_use]
    pub fn accepts(&self, agent_id: &str, envelope: &MessageEnvelope) -> bool {
        self.filters
            .read()
            .get(agent_id)
            .is_none_or(|filter| filter.accepts(envelope))
    }

    /// Returns the filtered agent ids, sorted.
    #[must_use]
    pub fn agents(&self) -> Vec<String> {
        self.filters.read().keys().cloned().collect()
    }

    /// Returns the number of installed filters.
    #[must_use]
    pub fn len(&self) -> usize {
        self.filters.read().len()
    }

    /// Returns `true` when no filter is installed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.filters.read().is_empty()
    }

    /// Removes every filter.
    pub fn clear(&self) {
        self.filters.write().clear();
    }
}
