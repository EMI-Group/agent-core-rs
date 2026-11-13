//! Typed team-event bus for the multi-agent runtime.
//!
//! Replaces Python's string-keyed callback framework (`AgentTeamEvents` in
//! `openjiuwen/core/runner/callback/events.py`) with a typed, synchronous bus.
//! Python triggers only the two receive events from
//! `team_runtime/message_router.py`; this module keeps those wire names for
//! parity and additionally covers membership, message, handoff, task, and error
//! events.
//!
//! [`TeamEventBus::emit`] stamps a strictly increasing sequence number and
//! invokes every matching listener before it returns, so delivery order equals
//! emission order on the emitting thread.
//! The bus holds no global state: a team owns one bus and shares it as an
//! `Arc`.
use std::collections::{BTreeSet, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use openjiuwen_common::utils::id::uuid_v4;
use parking_lot::{Mutex, RwLock};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
/// Scope prefix of the team event names, following Python's `scope:event` format.
pub const TEAM_EVENT_SCOPE: &str = "agent_team";
/// Python parity name for "an agent received a point-to-point call".
///
/// Python resolves `AgentTeamEvents.AGENT_P2P_RECEIVED` under the
/// `_framework` scope, so the fully scoped name keeps that prefix.
pub const AGENT_P2P_RECEIVED_EVENT: &str = "_framework:agent_p2p_received";
/// Python parity name for "an agent received a pub/sub call".
///
/// Python resolves `AgentTeamEvents.AGENT_PUBSUB_RECEIVED` under the
/// `_framework` scope, so the fully scoped name keeps that prefix.
pub const AGENT_PUBSUB_RECEIVED_EVENT: &str = "_framework:agent_pubsub_received";
/// Default number of most recent events kept in [`TeamEventBus`] history.
const DEFAULT_HISTORY_CAPACITY: usize = 256;
/// The kind of a team event.
///
/// Wire names are `snake_case`; [`TeamEventKind::event_name`] prefixes them with
/// the [`TEAM_EVENT_SCOPE`].
/// [`TeamEventKind::MessageReceived`] covers both point-to-point and pub/sub
/// receives: the presence of a topic id on the event distinguishes them.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum TeamEventKind {
    /// An agent joined the team.
    #[default]
    MemberJoined,
    /// An agent left the team.
    MemberLeft,
    /// An agent sent a message to a peer or to a topic.
    MessageSent,
    /// An agent received a message, either point-to-point or via pub/sub.
    MessageReceived,
    /// A handoff from one agent to another started.
    HandoffStarted,
    /// A handoff from one agent to another completed.
    HandoffCompleted,
    /// A task was assigned to an agent.
    TaskAssigned,
    /// An agent completed an assigned task.
    TaskCompleted,
    /// An agent or the team raised an error.
    Error,
}
impl TeamEventKind {
    /// Every kind, in declaration order.
    ///
    /// Handy for exhaustive filtering or for wiring one subscription per kind.
    pub const ALL: [Self; 9] = [
        Self::MemberJoined,
        Self::MemberLeft,
        Self::MessageSent,
        Self::MessageReceived,
        Self::HandoffStarted,
        Self::HandoffCompleted,
        Self::TaskAssigned,
        Self::TaskCompleted,
        Self::Error,
    ];
    /// Returns the unscoped `snake_case` wire name of the kind.
    ///
    /// For example `message_received` or `handoff_started`.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::MemberJoined => "member_joined",
            Self::MemberLeft => "member_left",
            Self::MessageSent => "message_sent",
            Self::MessageReceived => "message_received",
            Self::HandoffStarted => "handoff_started",
            Self::HandoffCompleted => "handoff_completed",
            Self::TaskAssigned => "task_assigned",
            Self::TaskCompleted => "task_completed",
            Self::Error => "error",
        }
    }
    /// Returns the scoped event name, for example `agent_team:member_joined`.
    #[must_use]
    pub fn event_name(&self) -> String {
        format!("{TEAM_EVENT_SCOPE}:{}", self.as_str())
    }
    /// Returns the Python `AgentTeamEvents` wire name for this kind, if any.
    ///
    /// Python separates a point-to-point receive
    /// ([`AGENT_P2P_RECEIVED_EVENT`]) from a pub/sub receive
    /// ([`AGENT_PUBSUB_RECEIVED_EVENT`]), while both map to
    /// [`TeamEventKind::MessageReceived`] here.
    /// A kind alone cannot tell them apart, so this method answers the
    /// point-to-point name for [`TeamEventKind::MessageReceived`] and `None` for
    /// every other kind; use [`TeamEvent::python_event_name`] when the event
    /// itself is at hand, because it can inspect the topic id.
    #[must_use]
    pub const fn python_event_name(&self) -> Option<&'static str> {
        match self {
            Self::MessageReceived => Some(AGENT_P2P_RECEIVED_EVENT),
            _ => None,
        }
    }
}
impl fmt::Display for TeamEventKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}
/// One event on a [`TeamEventBus`].
///
/// Every routing field is optional because a single event type serves
/// membership, messaging, handoff, task, and error notifications.
/// [`TeamEvent::agent_id`] names the subject agent (the actor for
/// [`TeamEventKind::MessageSent`], the receiver for
/// [`TeamEventKind::MessageReceived`]) and [`TeamEvent::peer_id`] the
/// counterpart.
/// The type round-trips through JSON, tolerates unknown fields, and treats every
/// field as optional on input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct TeamEvent {
    /// Unique event id, a hyphenated UUID v4 generated on construction.
    pub event_id: String,
    /// Delivery sequence stamped by [`TeamEventBus::emit`]; `0` before emission.
    pub sequence: u64,
    /// What happened.
    pub kind: TeamEventKind,
    /// Id of the team that produced the event.
    pub team_id: String,
    /// Optional team session id the event belongs to.
    pub session_id: Option<String>,
    /// Optional subject agent id.
    pub agent_id: Option<String>,
    /// Optional counterpart agent id.
    pub peer_id: Option<String>,
    /// Optional pub/sub topic id.
    pub topic_id: Option<String>,
    /// Optional message id, used to correlate a send with a receive.
    pub message_id: Option<String>,
    /// Free-form payload, always a JSON object.
    ///
    /// A missing field defaults to an empty object, and any non-object input
    /// (for example an explicit `null`) is read as an empty object.
    #[serde(default = "empty_detail", deserialize_with = "deserialize_detail")]
    pub detail: Value,
}
impl Default for TeamEvent {
    fn default() -> Self {
        Self::new(TeamEventKind::default(), String::new())
    }
}
impl TeamEvent {
    /// Creates an event of `kind` for `team_id` with a fresh id and no routing.
    #[must_use]
    pub fn new(kind: TeamEventKind, team_id: impl Into<String>) -> Self {
        Self {
            event_id: uuid_v4(),
            sequence: 0,
            kind,
            team_id: team_id.into(),
            session_id: None,
            agent_id: None,
            peer_id: None,
            topic_id: None,
            message_id: None,
            detail: empty_detail(),
        }
    }
    /// Attaches the team session id.
    #[must_use]
    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }
    /// Attaches the subject agent id.
    #[must_use]
    pub fn with_agent(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }
    /// Attaches the counterpart agent id.
    #[must_use]
    pub fn with_peer(mut self, peer_id: impl Into<String>) -> Self {
        self.peer_id = Some(peer_id.into());
        self
    }
    /// Attaches the pub/sub topic id.
    #[must_use]
    pub fn with_topic(mut self, topic_id: impl Into<String>) -> Self {
        self.topic_id = Some(topic_id.into());
        self
    }
    /// Attaches the message id.
    #[must_use]
    pub fn with_message_id(mut self, message_id: impl Into<String>) -> Self {
        self.message_id = Some(message_id.into());
        self
    }
    /// Inserts `key`/`value` into [`detail`](TeamEvent::detail).
    ///
    /// A detail that is not an object (for example one built by hand) is
    /// replaced with an empty object first, so the value stays a JSON object.
    #[must_use]
    pub fn with_detail(mut self, key: impl Into<String>, value: Value) -> Self {
        let key = key.into();
        if !self.detail.is_object() {
            self.detail = empty_detail();
        }
        if let Value::Object(map) = &mut self.detail {
            map.insert(key, value);
        }
        self
    }
    /// Creates a [`TeamEventKind::MemberJoined`] event for `agent_id`.
    #[must_use]
    pub fn member_joined(team_id: impl Into<String>, agent_id: impl Into<String>) -> Self {
        Self::new(TeamEventKind::MemberJoined, team_id).with_agent(agent_id)
    }
    /// Creates a [`TeamEventKind::MemberLeft`] event for `agent_id`.
    #[must_use]
    pub fn member_left(team_id: impl Into<String>, agent_id: impl Into<String>) -> Self {
        Self::new(TeamEventKind::MemberLeft, team_id).with_agent(agent_id)
    }
    /// Creates a [`TeamEventKind::MessageSent`] event from `sender` to `recipient`.
    #[must_use]
    pub fn message_sent(
        team_id: impl Into<String>,
        sender: impl Into<String>,
        recipient: impl Into<String>,
        message_id: impl Into<String>,
    ) -> Self {
        Self::new(TeamEventKind::MessageSent, team_id)
            .with_agent(sender)
            .with_peer(recipient)
            .with_message_id(message_id)
    }
    /// Creates a point-to-point [`TeamEventKind::MessageReceived`] event.
    ///
    /// `recipient` is the subject agent (the receiver) and `sender` the peer.
    #[must_use]
    pub fn message_received(
        team_id: impl Into<String>,
        sender: impl Into<String>,
        recipient: impl Into<String>,
        message_id: impl Into<String>,
    ) -> Self {
        Self::new(TeamEventKind::MessageReceived, team_id)
            .with_agent(recipient)
            .with_peer(sender)
            .with_message_id(message_id)
    }
    /// Creates a pub/sub [`TeamEventKind::MessageReceived`] event.
    ///
    /// `subscriber` is the subject agent and `sender` the peer; the topic id
    /// marks the receive as pub/sub rather than point-to-point.
    #[must_use]
    pub fn pubsub_received(
        team_id: impl Into<String>,
        sender: impl Into<String>,
        subscriber: impl Into<String>,
        topic_id: impl Into<String>,
        message_id: impl Into<String>,
    ) -> Self {
        Self::new(TeamEventKind::MessageReceived, team_id)
            .with_agent(subscriber)
            .with_peer(sender)
            .with_topic(topic_id)
            .with_message_id(message_id)
    }
    /// Creates a [`TeamEventKind::HandoffStarted`] event from `from_agent` to `to_agent`.
    #[must_use]
    pub fn handoff_started(
        team_id: impl Into<String>,
        from_agent: impl Into<String>,
        to_agent: impl Into<String>,
    ) -> Self {
        Self::new(TeamEventKind::HandoffStarted, team_id)
            .with_agent(from_agent)
            .with_peer(to_agent)
    }
    /// Creates a [`TeamEventKind::HandoffCompleted`] event from `from_agent` to `to_agent`.
    #[must_use]
    pub fn handoff_completed(
        team_id: impl Into<String>,
        from_agent: impl Into<String>,
        to_agent: impl Into<String>,
    ) -> Self {
        Self::new(TeamEventKind::HandoffCompleted, team_id)
            .with_agent(from_agent)
            .with_peer(to_agent)
    }
    /// Creates a [`TeamEventKind::TaskAssigned`] event for `agent_id`.
    ///
    /// The task description is recorded under the `task` detail key.
    #[must_use]
    pub fn task_assigned(
        team_id: impl Into<String>,
        agent_id: impl Into<String>,
        task: impl Into<String>,
    ) -> Self {
        Self::new(TeamEventKind::TaskAssigned, team_id)
            .with_agent(agent_id)
            .with_detail("task", Value::String(task.into()))
    }
    /// Creates a [`TeamEventKind::TaskCompleted`] event for `agent_id`.
    ///
    /// The task description is recorded under the `task` detail key.
    #[must_use]
    pub fn task_completed(
        team_id: impl Into<String>,
        agent_id: impl Into<String>,
        task: impl Into<String>,
    ) -> Self {
        Self::new(TeamEventKind::TaskCompleted, team_id)
            .with_agent(agent_id)
            .with_detail("task", Value::String(task.into()))
    }
    /// Creates an [`TeamEventKind::Error`] event for `agent_id`.
    ///
    /// The error message is recorded under the `error` detail key.
    #[must_use]
    pub fn error(
        team_id: impl Into<String>,
        agent_id: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self::new(TeamEventKind::Error, team_id)
            .with_agent(agent_id)
            .with_detail("error", Value::String(message.into()))
    }
    /// Returns the Python `AgentTeamEvents` wire name this event corresponds to.
    ///
    /// A [`MessageReceived`](TeamEventKind::MessageReceived) event carrying a
    /// [`topic_id`](TeamEvent::topic_id) is a pub/sub receive and maps to
    /// [`AGENT_PUBSUB_RECEIVED_EVENT`]; a receive without a topic maps to
    /// [`AGENT_P2P_RECEIVED_EVENT`].
    /// Every other kind has no Python counterpart and answers `None`.
    #[must_use]
    pub fn python_event_name(&self) -> Option<&'static str> {
        match self.kind {
            TeamEventKind::MessageReceived if self.topic_id.is_some() => {
                Some(AGENT_PUBSUB_RECEIVED_EVENT)
            }
            kind => kind.python_event_name(),
        }
    }
}
/// Which events a [`TeamEventBus`] subscription receives.
///
/// Every field defaults to `None`, which matches everything, so
/// [`TeamEventFilter::new`] and [`TeamEventFilter::all`] are equivalent.
/// A filter is evaluated per subscriber at dispatch time, so it can be reused
/// across subscriptions and cloned freely.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct TeamEventFilter {
    /// Optional set of accepted kinds; `None` accepts every kind.
    pub kinds: Option<BTreeSet<TeamEventKind>>,
    /// Optional exact team id; `None` accepts every team.
    pub team_id: Option<String>,
    /// Optional exact session id; `None` accepts events without a session too.
    pub session_id: Option<String>,
    /// Optional accepted agent ids; `None` accepts every agent.
    ///
    /// An id matches when it is either the subject agent
    /// ([`TeamEvent::agent_id`]) or the counterpart ([`TeamEvent::peer_id`]).
    pub agent_ids: Option<BTreeSet<String>>,
}
impl TeamEventFilter {
    /// Creates a filter that matches every event.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Creates a filter that matches every event.
    #[must_use]
    pub fn all() -> Self {
        Self::default()
    }
    /// Adds one accepted kind.
    ///
    /// Repeated calls accumulate; a filter with several kinds accepts any of
    /// them.
    #[must_use]
    pub fn kind(mut self, kind: TeamEventKind) -> Self {
        self.kinds.get_or_insert_default().insert(kind);
        self
    }
    /// Adds every kind yielded by `kinds`.
    #[must_use]
    pub fn kinds(mut self, kinds: impl IntoIterator<Item = TeamEventKind>) -> Self {
        self.kinds.get_or_insert_default().extend(kinds);
        self
    }
    /// Restricts the filter to one team id.
    #[must_use]
    pub fn team(mut self, team_id: impl Into<String>) -> Self {
        self.team_id = Some(team_id.into());
        self
    }
    /// Restricts the filter to one session id.
    #[must_use]
    pub fn session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }
    /// Adds one accepted agent id.
    ///
    /// Repeated calls accumulate; see [`agent_ids`](TeamEventFilter::agent_ids)
    /// for the peer matching rule.
    #[must_use]
    pub fn agent(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_ids
            .get_or_insert_default()
            .insert(agent_id.into());
        self
    }
    /// Returns `true` when `event` satisfies every constraint of this filter.
    ///
    /// A session constraint only matches events that carry that exact session id,
    /// and an agent constraint matches the subject agent or the counterpart.
    #[must_use]
    pub fn matches(&self, event: &TeamEvent) -> bool {
        if let Some(kinds) = &self.kinds {
            if !kinds.contains(&event.kind) {
                return false;
            }
        }
        if let Some(team_id) = &self.team_id {
            if event.team_id != *team_id {
                return false;
            }
        }
        if let Some(session_id) = &self.session_id {
            if event.session_id.as_deref() != Some(session_id.as_str()) {
                return false;
            }
        }
        if let Some(agent_ids) = &self.agent_ids {
            let subject = event.agent_id.as_deref();
            let peer = event.peer_id.as_deref();
            let matched = subject.is_some_and(|id| agent_ids.contains(id))
                || peer.is_some_and(|id| agent_ids.contains(id));
            if !matched {
                return false;
            }
        }
        true
    }
}
/// A listener invoked synchronously for every event that matches its filter.
///
/// Listeners must be `Send + Sync` because one bus is shared across threads, and
/// a listener may observe concurrent calls from different emitters.
pub type TeamEventListener = Arc<dyn Fn(&TeamEvent) + Send + Sync>;
/// One registered listener: its id, filter, and callback.
type Subscription = (u64, TeamEventFilter, TeamEventListener);
/// A typed event bus with bounded history and filtered subscriptions.
///
/// The bus is shared as an `Arc` (it is not `Clone`); construct it with
/// [`TeamEventBus::new`] or [`TeamEventBus::with_capacity`].
/// Emission is synchronous: [`TeamEventBus::emit`] stamps the sequence number,
/// records the event in the bounded history, and calls every matching listener in
/// ascending subscription-id order before returning.
/// No lock is held while a listener runs, so a listener may emit or
/// (un)subscribe on the same bus.
pub struct TeamEventBus {
    /// Registered listeners, ordered by ascending subscription id.
    subscriptions: RwLock<Vec<Subscription>>,
    /// Next subscription id; ids start at `1`.
    next_subscription_id: AtomicU64,
    /// Sequence number of the most recent emission; `0` before the first emit.
    sequence: AtomicU64,
    /// Maximum number of events kept in [`history`](TeamEventBus::history).
    history_capacity: usize,
    /// Retained events, ordered by sequence and bounded by `history_capacity`.
    history: Mutex<VecDeque<TeamEvent>>,
}
impl TeamEventBus {
    /// Creates a bus that keeps the 256 most recent events in its history.
    #[must_use]
    pub fn new() -> Arc<Self> {
        Self::with_capacity(DEFAULT_HISTORY_CAPACITY)
    }
    /// Creates a bus keeping at most `history_capacity` recent events.
    ///
    /// A capacity of `0` keeps no history: [`history`](TeamEventBus::history)
    /// stays empty while emission and delivery are unaffected.
    #[must_use]
    pub fn with_capacity(history_capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            subscriptions: RwLock::new(Vec::new()),
            next_subscription_id: AtomicU64::new(1),
            sequence: AtomicU64::new(0),
            history_capacity,
            history: Mutex::new(VecDeque::new()),
        })
    }
    /// Registers `listener` for events accepted by `filter`.
    ///
    /// Returns the subscription id (ascending from `1`), which
    /// [`unsubscribe`](TeamEventBus::unsubscribe) takes.
    pub fn subscribe(&self, filter: TeamEventFilter, listener: TeamEventListener) -> u64 {
        let subscription_id = self.next_subscription_id.fetch_add(1, Ordering::Relaxed);
        self.subscriptions
            .write()
            .push((subscription_id, filter, listener));
        subscription_id
    }
    /// Removes the subscription with `subscription_id`.
    ///
    /// Returns `true` when a subscription was removed and `false` for an unknown
    /// id; the call never panics.
    #[must_use]
    pub fn unsubscribe(&self, subscription_id: u64) -> bool {
        let mut subscriptions = self.subscriptions.write();
        let before = subscriptions.len();
        subscriptions.retain(|(id, _, _)| *id != subscription_id);
        subscriptions.len() != before
    }
    /// Returns the number of registered subscriptions.
    #[must_use]
    pub fn subscriber_count(&self) -> usize {
        self.subscriptions.read().len()
    }
    /// Delivers `event` to every matching listener and returns the stamped event.
    ///
    /// The sequence number is assigned from a process-wide-per-bus counter
    /// starting at `1` and is strictly increasing across emitters.
    /// Listeners run synchronously on the calling thread in ascending
    /// subscription-id order, so a panicking listener propagates out of `emit`
    /// (there is no `catch_unwind`); listeners must not panic.
    pub fn emit(&self, mut event: TeamEvent) -> TeamEvent {
        let sequence = self.sequence.fetch_add(1, Ordering::SeqCst) + 1;
        event.sequence = sequence;
        self.record(&event);
        // Clone the subscription arcs, then drop the lock before calling out: a
        // listener may subscribe, unsubscribe, or emit re-entrantly.
        let listeners: Vec<(TeamEventFilter, TeamEventListener)> = {
            let subscriptions = self.subscriptions.read();
            subscriptions
                .iter()
                .map(|(_, filter, listener)| (filter.clone(), Arc::clone(listener)))
                .collect()
        };
        for (filter, listener) in &listeners {
            if filter.matches(&event) {
                listener(&event);
            }
        }
        event
    }
    /// Returns a snapshot of the retained history, oldest (lowest sequence) first.
    #[must_use]
    pub fn history(&self) -> Vec<TeamEvent> {
        self.history.lock().iter().cloned().collect()
    }
    /// Returns the retained history entries of `kind`, oldest (lowest sequence) first.
    #[must_use]
    pub fn history_of(&self, kind: TeamEventKind) -> Vec<TeamEvent> {
        self.history
            .lock()
            .iter()
            .filter(|event| event.kind == kind)
            .cloned()
            .collect()
    }
    /// Returns the sequence number of the most recent emission, or `0` if none.
    #[must_use]
    pub fn last_sequence(&self) -> u64 {
        self.sequence.load(Ordering::SeqCst)
    }
    /// Inserts `event` into the bounded history in sequence order.
    ///
    /// The sequence number is assigned before the history lock is taken, so
    /// concurrent emitters can arrive out of order; inserting at the sequence
    /// position keeps the retained window ordered by sequence and makes the
    /// oldest (lowest-sequence) entries the ones dropped first.
    fn record(&self, event: &TeamEvent) {
        if self.history_capacity == 0 {
            return;
        }
        let mut history = self.history.lock();
        let position = history.partition_point(|existing| existing.sequence < event.sequence);
        history.insert(position, event.clone());
        while history.len() > self.history_capacity {
            history.pop_front();
        }
    }
}
impl fmt::Debug for TeamEventBus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TeamEventBus")
            .field("subscriber_count", &self.subscriber_count())
            .field("history_capacity", &self.history_capacity)
            .field("last_sequence", &self.last_sequence())
            .finish_non_exhaustive()
    }
}
/// Returns an empty JSON object, the default for [`TeamEvent::detail`].
fn empty_detail() -> Value {
    Value::Object(Map::new())
}
/// Reads a detail value, mapping any non-object input to an empty object.
fn deserialize_detail<'de, D>(deserializer: D) -> std::result::Result<Value, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    Ok(match value {
        Value::Object(map) => Value::Object(map),
        _ => empty_detail(),
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    /// Creates a listener that records every delivered sequence number.
    fn recording_listener() -> (TeamEventListener, Arc<Mutex<Vec<u64>>>) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&seen);
        let listener: TeamEventListener = Arc::new(move |event: &TeamEvent| {
            sink.lock().push(event.sequence);
        });
        (listener, seen)
    }
    #[test]
    fn all_kinds_expose_stable_wire_names() {
        assert_eq!(TeamEventKind::ALL.len(), 9);
        let names: Vec<&str> = TeamEventKind::ALL
            .iter()
            .map(TeamEventKind::as_str)
            .collect();
        assert_eq!(
            names,
            [
                "member_joined",
                "member_left",
                "message_sent",
                "message_received",
                "handoff_started",
                "handoff_completed",
                "task_assigned",
                "task_completed",
                "error",
            ]
        );
        assert_eq!(
            TeamEventKind::MemberJoined.event_name(),
            "agent_team:member_joined"
        );
        assert_eq!(
            TeamEventKind::MessageReceived.to_string(),
            "message_received"
        );
    }
    #[test]
    fn python_names_keep_python_parity() {
        assert_eq!(
            TeamEventKind::MessageReceived.python_event_name(),
            Some(AGENT_P2P_RECEIVED_EVENT)
        );
        assert_eq!(TeamEventKind::MemberJoined.python_event_name(), None);
        let p2p = TeamEvent::message_received("team", "a", "b", "m-1");
        assert_eq!(p2p.python_event_name(), Some(AGENT_P2P_RECEIVED_EVENT));
        let pubsub = TeamEvent::pubsub_received("team", "a", "b", "topic", "m-2");
        assert_eq!(
            pubsub.python_event_name(),
            Some(AGENT_PUBSUB_RECEIVED_EVENT)
        );
        assert_eq!(
            TeamEvent::member_left("team", "a").python_event_name(),
            None
        );
    }
    #[test]
    fn constructors_populate_routing_fields() {
        let event = TeamEvent::member_joined("team", "alpha");
        assert!(uuid::Uuid::parse_str(&event.event_id).is_ok());
        assert_eq!(event.sequence, 0);
        assert_eq!(event.kind, TeamEventKind::MemberJoined);
        assert_eq!(event.team_id, "team");
        assert_eq!(event.agent_id.as_deref(), Some("alpha"));
        assert_eq!(event.detail, serde_json::json!({}));
        let sent = TeamEvent::message_sent("team", "alpha", "beta", "m-1");
        assert_eq!(sent.agent_id.as_deref(), Some("alpha"));
        assert_eq!(sent.peer_id.as_deref(), Some("beta"));
        assert_eq!(sent.message_id.as_deref(), Some("m-1"));
        let received = TeamEvent::message_received("team", "alpha", "beta", "m-1");
        assert_eq!(received.agent_id.as_deref(), Some("beta"));
        assert_eq!(received.peer_id.as_deref(), Some("alpha"));
        let pubsub = TeamEvent::pubsub_received("team", "alpha", "beta", "topic", "m-2");
        assert_eq!(pubsub.topic_id.as_deref(), Some("topic"));
        assert_eq!(pubsub.agent_id.as_deref(), Some("beta"));
        let handoff = TeamEvent::handoff_started("team", "alpha", "beta");
        assert_eq!(handoff.kind, TeamEventKind::HandoffStarted);
        assert_eq!(handoff.peer_id.as_deref(), Some("beta"));
        let done = TeamEvent::handoff_completed("team", "alpha", "beta");
        assert_eq!(done.kind, TeamEventKind::HandoffCompleted);
        let assigned = TeamEvent::task_assigned("team", "alpha", "write tests");
        assert_eq!(assigned.detail, serde_json::json!({"task": "write tests"}));
        let completed = TeamEvent::task_completed("team", "alpha", "write tests");
        assert_eq!(completed.detail, serde_json::json!({"task": "write tests"}));
        let error = TeamEvent::error("team", "alpha", "boom");
        assert_eq!(error.detail, serde_json::json!({"error": "boom"}));
    }
    #[test]
    fn with_detail_keeps_detail_an_object() {
        let mut odd = TeamEvent::new(TeamEventKind::Error, "team");
        odd.detail = Value::Null;
        let event = odd.with_detail("key", serde_json::json!(1));
        assert_eq!(event.detail, serde_json::json!({"key": 1}));
    }
    #[test]
    fn emit_stamps_increasing_sequences() {
        let bus = TeamEventBus::new();
        assert_eq!(bus.last_sequence(), 0);
        let first = bus.emit(TeamEvent::member_joined("team", "alpha"));
        let second = bus.emit(TeamEvent::member_joined("team", "beta"));
        assert_eq!(first.sequence, 1);
        assert_eq!(second.sequence, 2);
        assert_eq!(bus.last_sequence(), 2);
    }
    #[test]
    fn listeners_run_in_subscription_order() {
        let bus = TeamEventBus::new();
        let order = Arc::new(Mutex::new(Vec::new()));
        for id in ["first", "second"] {
            let order = Arc::clone(&order);
            bus.subscribe(
                TeamEventFilter::new(),
                Arc::new(move |_event: &TeamEvent| order.lock().push(id)),
            );
        }
        assert_eq!(bus.subscriber_count(), 2);
        bus.emit(TeamEvent::member_joined("team", "alpha"));
        assert_eq!(*order.lock(), ["first", "second"]);
    }
    #[test]
    fn filters_select_kind_team_session_and_agents() {
        let bus = TeamEventBus::new();
        let (kind_listener, kind_seen) = recording_listener();
        let (team_listener, team_seen) = recording_listener();
        let (session_listener, session_seen) = recording_listener();
        let (agent_listener, agent_seen) = recording_listener();
        bus.subscribe(
            TeamEventFilter::new().kind(TeamEventKind::HandoffStarted),
            kind_listener,
        );
        bus.subscribe(TeamEventFilter::new().team("other"), team_listener);
        bus.subscribe(TeamEventFilter::new().session("s-1"), session_listener);
        bus.subscribe(TeamEventFilter::new().agent("beta"), agent_listener);
        bus.emit(TeamEvent::handoff_started("team", "alpha", "beta").with_session("s-1"));
        bus.emit(TeamEvent::member_left("team", "gamma"));
        assert_eq!(*kind_seen.lock(), [1]);
        assert_eq!(*team_seen.lock(), Vec::<u64>::new());
        assert_eq!(*session_seen.lock(), [1]);
        // The second event has no peer, so the agent filter drops it; the first
        // matches through the peer id.
        assert_eq!(*agent_seen.lock(), [1]);
    }
    #[test]
    fn filter_matches_everything_by_default() {
        let filter = TeamEventFilter::all();
        assert!(filter.matches(&TeamEvent::error("team", "alpha", "boom")));
        assert_eq!(TeamEventFilter::new(), TeamEventFilter::default());
    }
    #[test]
    fn unsubscribe_stops_delivery() {
        let bus = TeamEventBus::new();
        let (first, first_seen) = recording_listener();
        let (second, second_seen) = recording_listener();
        let first_id = bus.subscribe(TeamEventFilter::new(), first);
        let second_id = bus.subscribe(TeamEventFilter::new(), second);
        assert!(first_id < second_id);
        assert!(bus.unsubscribe(first_id));
        assert!(!bus.unsubscribe(first_id));
        assert!(!bus.unsubscribe(9_999));
        assert_eq!(bus.subscriber_count(), 1);
        bus.emit(TeamEvent::member_joined("team", "alpha"));
        assert_eq!(*first_seen.lock(), Vec::<u64>::new());
        assert_eq!(*second_seen.lock(), [1]);
    }
    #[test]
    fn history_is_bounded_and_oldest_first() {
        let bus = TeamEventBus::with_capacity(3);
        for index in 0..5 {
            bus.emit(TeamEvent::member_joined("team", format!("agent-{index}")));
        }
        let history = bus.history();
        assert_eq!(history.len(), 3);
        let sequences: Vec<u64> = history.iter().map(|event| event.sequence).collect();
        assert_eq!(sequences, [3, 4, 5]);
        let joined = bus.history_of(TeamEventKind::MemberJoined);
        assert_eq!(joined.len(), 3);
        assert_eq!(
            bus.history_of(TeamEventKind::Error),
            Vec::<TeamEvent>::new()
        );
    }
    #[test]
    fn zero_capacity_keeps_no_history() {
        let bus = TeamEventBus::with_capacity(0);
        let (listener, seen) = recording_listener();
        bus.subscribe(TeamEventFilter::new(), listener);
        bus.emit(TeamEvent::member_joined("team", "alpha"));
        assert_eq!(bus.history(), Vec::<TeamEvent>::new());
        assert_eq!(*seen.lock(), [1]);
    }
    #[test]
    fn serde_round_trip_and_unknown_fields() {
        let event = TeamEvent::handoff_completed("team", "alpha", "beta")
            .with_session("s-1")
            .with_message_id("m-1")
            .with_detail("hops", serde_json::json!(2));
        let wire = serde_json::to_value(&event).expect("event serializes");
        assert_eq!(wire["kind"], "handoff_completed");
        let decoded: TeamEvent = serde_json::from_value(wire).expect("event deserializes");
        assert_eq!(decoded, event);
        let minimal: TeamEvent = serde_json::from_value(serde_json::json!({
            "kind": "handoff_started",
            "team_id": "team",
            "unknown": true,
        }))
        .expect("unknown fields are tolerated");
        assert_eq!(minimal.kind, TeamEventKind::HandoffStarted);
        assert_eq!(minimal.sequence, 0);
        assert_eq!(minimal.detail, serde_json::json!({}));
        let null_detail: TeamEvent = serde_json::from_value(serde_json::json!({
            "kind": "error",
            "detail": null,
        }))
        .expect("null detail is tolerated");
        assert_eq!(null_detail.detail, serde_json::json!({}));
    }
    #[test]
    fn schema_exposes_snake_case_properties() {
        let schema = serde_json::to_value(schemars::schema_for!(TeamEvent))
            .expect("event schema serializes");
        let mut properties: Vec<String> = schema["properties"]
            .as_object()
            .expect("a struct schema has properties")
            .keys()
            .cloned()
            .collect();
        properties.sort();
        assert_eq!(
            properties,
            [
                "agent_id",
                "detail",
                "event_id",
                "kind",
                "message_id",
                "peer_id",
                "sequence",
                "session_id",
                "team_id",
                "topic_id",
            ]
        );
        let filter = serde_json::to_value(schemars::schema_for!(TeamEventFilter))
            .expect("filter schema serializes");
        let mut filter_properties: Vec<String> = filter["properties"]
            .as_object()
            .expect("a struct schema has properties")
            .keys()
            .cloned()
            .collect();
        filter_properties.sort();
        assert_eq!(
            filter_properties,
            ["agent_ids", "kinds", "session_id", "team_id"]
        );
    }
    #[test]
    fn debug_reports_state_without_listeners() {
        let bus = TeamEventBus::with_capacity(8);
        bus.subscribe(TeamEventFilter::new(), Arc::new(|_event: &TeamEvent| {}));
        bus.emit(TeamEvent::member_joined("team", "alpha"));
        let rendered = format!("{bus:?}");
        assert!(rendered.contains("subscriber_count: 1"), "{rendered}");
        assert!(rendered.contains("history_capacity: 8"), "{rendered}");
        assert!(rendered.contains("last_sequence: 1"), "{rendered}");
    }
}
