//! The in-process transport: one per-instance handler registry.
//!
//! Python parity source: `openjiuwen/agent_teams/messager/inprocess.py` (the `_Bus` handler bus).
//! Python keeps a single process-global `_Bus`; this port deliberately makes the bus per
//! [`InProcessMessager`] so two teams (or two tests) never see each other's subscriptions.
//!
//! Delivery rules pinned by `src/messager/CONTEXT.md`:
//!
//! - the registry is guarded by a [`parking_lot::RwLock`] and a sink list is cloned before any
//!   sink runs, so a sink may subscribe or unsubscribe re-entrantly without deadlocking;
//! - publish fan-out is in sorted member order (`BTreeMap`), send hits exactly one direct handler;
//! - a panicking sink is caught, logged, and never fails the publish/send nor blocks its peers;
//! - an empty `sender_id` is stamped with this messager's sender; an explicit one is preserved;
//! - a stopped messager drops events with a warning but keeps its subscriptions, so `start()`
//!   resumes delivery.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::RwLock;

use openjiuwen_common::error::Result;

use super::factory::InProcessMessagerConfig;
use super::{EventSink, Messager};
use crate::schema::{EventMessage, TeamTopic};

/// Transport kind reported by [`Messager::name`] (Python `"inprocess"`).
const TRANSPORT_NAME: &str = "inprocess";

/// The mutable half of the messager, guarded by one lock.
///
/// `topics` maps a topic to its subscribers keyed by member name (sorted by construction),
/// `direct` holds one point-to-point handler per member, and `teams` mirrors Python's
/// `subscribe_transport` bookkeeping.
#[derive(Default)]
struct Registry {
    /// Per-topic subscribers, sorted by member name.
    topics: HashMap<TeamTopic, BTreeMap<String, EventSink>>,
    /// One point-to-point handler per member name.
    direct: BTreeMap<String, EventSink>,
    /// Team names this messager was asked to serve (sorted, de-duplicated).
    teams: BTreeSet<String>,
    /// `true` while events are delivered.
    started: bool,
}

/// Lock-guarded snapshot of the registry, taken while the lifecycle flag is checked.
///
/// Returning this instead of a bare `Option` keeps the two outcomes distinct at the call site: a
/// stopped messager logs "dropped", an empty handler set logs nothing.
enum Snapshot<T> {
    /// The messager is stopped; the event is dropped with a warning.
    Stopped,
    /// The messager is live; the value is the cloned handler set (possibly empty).
    Live(T),
}

/// In-process [`Messager`]: a per-instance publish/subscribe and point-to-point registry.
///
/// The messager is live as soon as it is constructed (`is_started() == true`), matching Python's
/// handler bus, so callers do not have to call `start()` before publishing.
pub struct InProcessMessager {
    /// Member name stamped into events that arrive with an empty `sender_id`.
    sender: String,
    /// Handler registry; never held across a sink invocation.
    registry: RwLock<Registry>,
}

impl InProcessMessager {
    /// Builds a live messager with an empty sender (no stamping).
    #[must_use]
    pub fn new() -> Self {
        Self::with_sender("")
    }

    /// Builds a live messager that stamps `sender` into events published with an empty `sender_id`.
    #[must_use]
    pub fn with_sender(sender: impl Into<String>) -> Self {
        Self {
            sender: sender.into(),
            registry: RwLock::new(Registry {
                started: true,
                ..Registry::default()
            }),
        }
    }

    /// Builds a messager from the Python `InProcessMessagerConfig`.
    ///
    /// `peers` belongs to the `pyzmq` backend (address bookkeeping); the in-process transport
    /// addresses members by name and ignores it.
    #[must_use]
    pub fn from_config(config: &InProcessMessagerConfig) -> Self {
        Self::with_sender(config.sender.clone())
    }

    /// Returns the member name this messager stamps into unaddressed events.
    #[must_use]
    pub fn sender(&self) -> &str {
        &self.sender
    }

    /// Returns the recorded team names, sorted and de-duplicated.
    #[must_use]
    pub fn team_subscriptions(&self) -> Vec<String> {
        self.registry.read().teams.iter().cloned().collect()
    }

    /// Returns `event` with this messager's sender stamped when the producer left it empty.
    fn stamp(&self, mut event: EventMessage) -> EventMessage {
        if event.sender_id.is_empty() {
            event.sender_id.clone_from(&self.sender);
        }
        event
    }

    /// Clones the subscriber list of `topic` under a read lock, or reports the messager stopped.
    fn topic_snapshot(&self, topic: TeamTopic) -> Snapshot<Vec<(String, EventSink)>> {
        let registry = self.registry.read();
        if !registry.started {
            return Snapshot::Stopped;
        }
        let sinks = registry
            .topics
            .get(&topic)
            .map(|sinks| {
                sinks
                    .iter()
                    .map(|(member, sink)| (member.clone(), Arc::clone(sink)))
                    .collect()
            })
            .unwrap_or_default();
        Snapshot::Live(sinks)
    }

    /// Clones the direct handler of `target` under a read lock, or reports the messager stopped.
    fn direct_snapshot(&self, target: &str) -> Snapshot<Option<EventSink>> {
        let registry = self.registry.read();
        if !registry.started {
            return Snapshot::Stopped;
        }
        Snapshot::Live(registry.direct.get(target).map(Arc::clone))
    }
}

impl Default for InProcessMessager {
    /// Same as [`InProcessMessager::new`]: live, empty sender.
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for InProcessMessager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let registry = self.registry.read();
        f.debug_struct("InProcessMessager")
            .field("sender", &self.sender)
            .field("started", &registry.started)
            .field("topics", &registry.topics.len())
            .field("direct_handlers", &registry.direct.len())
            .field("teams", &registry.teams)
            .finish()
    }
}

#[async_trait]
impl Messager for InProcessMessager {
    fn name(&self) -> &'static str {
        TRANSPORT_NAME
    }

    fn subscribe(&self, member: &str, topics: &[TeamTopic], sink: EventSink) -> Result<()> {
        let mut registry = self.registry.write();
        for topic in topics {
            registry
                .topics
                .entry(*topic)
                .or_default()
                .insert(member.to_owned(), Arc::clone(&sink));
        }
        // Python registers the same handler for direct delivery in the same call.
        if !topics.is_empty() {
            registry.direct.insert(member.to_owned(), sink);
        }
        Ok(())
    }

    fn unsubscribe(&self, member: &str) -> Result<()> {
        let mut registry = self.registry.write();
        registry.topics.retain(|_, sinks| {
            sinks.remove(member);
            !sinks.is_empty()
        });
        registry.direct.remove(member);
        Ok(())
    }

    async fn publish(&self, topic: TeamTopic, event: EventMessage) -> Result<()> {
        let sinks = match self.topic_snapshot(topic) {
            Snapshot::Stopped => {
                tracing::warn!(
                    transport = TRANSPORT_NAME,
                    topic = topic.as_str(),
                    event_type = %event.event_type,
                    "message dropped: messager is stopped"
                );
                return Ok(());
            }
            Snapshot::Live(sinks) => sinks,
        };
        let event = self.stamp(event);
        for (member, sink) in sinks {
            deliver(&member, &sink, &event);
        }
        Ok(())
    }

    async fn send(&self, target: &str, event: EventMessage) -> Result<()> {
        let sink = match self.direct_snapshot(target) {
            Snapshot::Stopped => {
                tracing::warn!(
                    transport = TRANSPORT_NAME,
                    target,
                    event_type = %event.event_type,
                    "message dropped: messager is stopped"
                );
                return Ok(());
            }
            Snapshot::Live(sink) => sink,
        };
        let event = self.stamp(event);
        if let Some(sink) = sink {
            deliver(target, &sink, &event);
        }
        Ok(())
    }

    async fn subscribe_transport(&self, team_name: &str) -> Result<()> {
        self.registry.write().teams.insert(team_name.to_owned());
        Ok(())
    }

    fn subscribers(&self, topic: TeamTopic) -> Vec<String> {
        self.registry
            .read()
            .topics
            .get(&topic)
            .map(|sinks| sinks.keys().cloned().collect())
            .unwrap_or_default()
    }

    fn is_started(&self) -> bool {
        self.registry.read().started
    }

    async fn start(&self) -> Result<()> {
        self.registry.write().started = true;
        Ok(())
    }

    async fn stop(&self) -> Result<()> {
        self.registry.write().started = false;
        Ok(())
    }
}

/// Invokes one sink, containing a panic so the remaining subscribers still receive the event.
///
/// `AssertUnwindSafe` is sound here because the closure only touches a clone of the envelope: a
/// panicking sink cannot leave shared state half-updated for the next one.
fn deliver(member: &str, sink: &EventSink, event: &EventMessage) {
    let outcome = catch_unwind(AssertUnwindSafe(|| sink(event.clone())));
    if outcome.is_err() {
        tracing::warn!(
            transport = TRANSPORT_NAME,
            member,
            event_type = %event.event_type,
            "message dropped for this subscriber: sink panicked"
        );
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use futures::executor::block_on;
    use parking_lot::Mutex;
    use serde_json::json;

    use super::*;
    use crate::messager::factory::MessagerPeerConfig;

    /// Shared transcript of every event a recording sink observed.
    type Log = Arc<Mutex<Vec<(String, EventMessage)>>>;

    fn new_log() -> Log {
        Arc::new(Mutex::new(Vec::new()))
    }

    fn recorder(log: &Log, label: &str) -> EventSink {
        let log = Arc::clone(log);
        let label = label.to_owned();
        Arc::new(move |event| log.lock().push((label.clone(), event)))
    }

    fn panicker() -> EventSink {
        Arc::new(|_event| panic!("sink exploded"))
    }

    fn observed(log: &Log) -> Vec<(String, EventMessage)> {
        log.lock().clone()
    }

    fn observed_members(log: &Log) -> Vec<String> {
        log.lock().iter().map(|(label, _)| label.clone()).collect()
    }

    #[test]
    fn fan_out_is_sorted_by_member_name_and_stable_across_repeats() {
        let bus = InProcessMessager::with_sender("leader");
        let log = new_log();
        for member in ["carol", "alice", "bob"] {
            bus.subscribe(member, &[TeamTopic::Team], recorder(&log, member))
                .unwrap();
        }
        let event = EventMessage::new("member_status_changed", json!({"status": "ready"}));
        block_on(bus.publish(TeamTopic::Team, event.clone())).unwrap();
        block_on(bus.publish(TeamTopic::Team, event)).unwrap();
        assert_eq!(
            observed_members(&log),
            ["alice", "bob", "carol", "alice", "bob", "carol"]
        );
        assert_eq!(bus.subscribers(TeamTopic::Team), ["alice", "bob", "carol"]);
    }

    #[test]
    fn a_member_with_two_topics_receives_once_per_matching_publish() {
        let bus = InProcessMessager::new();
        let log = new_log();
        bus.subscribe(
            "bob",
            &[TeamTopic::Team, TeamTopic::Message],
            recorder(&log, "bob"),
        )
        .unwrap();
        assert_eq!(bus.subscribers(TeamTopic::Team), ["bob"]);
        assert_eq!(bus.subscribers(TeamTopic::Message), ["bob"]);
        assert!(bus.subscribers(TeamTopic::Task).is_empty());

        block_on(bus.publish(TeamTopic::Team, EventMessage::new("team", json!({})))).unwrap();
        block_on(bus.publish(TeamTopic::Message, EventMessage::new("message", json!({})))).unwrap();
        block_on(bus.publish(TeamTopic::Task, EventMessage::new("task", json!({})))).unwrap();
        assert_eq!(observed_members(&log), ["bob", "bob"]);
        assert_eq!(observed(&log)[1].1.event_type, "message");
    }

    #[test]
    fn empty_sender_is_stamped_and_explicit_sender_is_preserved() {
        let bus = InProcessMessager::with_sender("leader");
        let log = new_log();
        bus.subscribe("bob", &[TeamTopic::Message], recorder(&log, "bob"))
            .unwrap();

        block_on(bus.publish(TeamTopic::Message, EventMessage::new("message", json!({})))).unwrap();
        block_on(bus.publish(
            TeamTopic::Message,
            EventMessage::new("message", json!({})).with_sender("carol"),
        ))
        .unwrap();
        block_on(bus.send("bob", EventMessage::new("message", json!({})))).unwrap();

        let observed = observed(&log);
        assert_eq!(observed[0].1.sender_id, "leader");
        assert_eq!(observed[1].1.sender_id, "carol");
        // `send` carries the message but `bob` only holds `Message`; the send is still stamped.
        assert_eq!(observed[2].1.sender_id, "leader");
    }

    #[test]
    fn a_messager_without_a_sender_leaves_an_empty_sender_empty() {
        let bus = InProcessMessager::new();
        assert_eq!(bus.sender(), "");
        let log = new_log();
        bus.subscribe("bob", &[TeamTopic::Team], recorder(&log, "bob"))
            .unwrap();
        block_on(bus.publish(TeamTopic::Team, EventMessage::new("team", json!({})))).unwrap();
        assert_eq!(observed(&log)[0].1.sender_id, "");
    }

    #[test]
    fn send_reaches_only_the_target_member() {
        let bus = InProcessMessager::new();
        let log = new_log();
        for member in ["alice", "bob"] {
            bus.subscribe(member, &[TeamTopic::Message], recorder(&log, member))
                .unwrap();
        }
        block_on(bus.send("bob", EventMessage::new("message", json!({})))).unwrap();
        assert_eq!(observed_members(&log), ["bob"]);

        // Unknown target, a member subscribed only to topics of a different kind: both no-ops.
        block_on(bus.send("ghost", EventMessage::new("message", json!({})))).unwrap();
        assert_eq!(observed_members(&log), ["bob"]);
    }

    #[test]
    fn subscribe_without_topics_installs_no_direct_handler() {
        let bus = InProcessMessager::new();
        let log = new_log();
        bus.subscribe("eve", &[], recorder(&log, "eve")).unwrap();
        assert!(bus.subscribers(TeamTopic::Team).is_empty());
        block_on(bus.send("eve", EventMessage::new("message", json!({})))).unwrap();
        assert!(observed(&log).is_empty());
    }

    #[test]
    fn duplicate_subscribe_invokes_one_sink_and_resubscribe_replaces_it() {
        let bus = InProcessMessager::new();
        let log = new_log();
        bus.subscribe("bob", &[TeamTopic::Team], recorder(&log, "first"))
            .unwrap();
        bus.subscribe("bob", &[TeamTopic::Team], recorder(&log, "second"))
            .unwrap();
        block_on(bus.publish(TeamTopic::Team, EventMessage::new("team", json!({})))).unwrap();
        assert_eq!(observed_members(&log), ["second"]);
    }

    #[test]
    fn unsubscribe_stops_delivery_and_is_idempotent() {
        let bus = InProcessMessager::new();
        let log = new_log();
        bus.subscribe("bob", &[TeamTopic::Team], recorder(&log, "bob"))
            .unwrap();
        bus.unsubscribe("bob").unwrap();
        bus.unsubscribe("bob").unwrap();
        bus.unsubscribe("ghost").unwrap();

        assert!(bus.subscribers(TeamTopic::Team).is_empty());
        block_on(bus.publish(TeamTopic::Team, EventMessage::new("team", json!({})))).unwrap();
        block_on(bus.send("bob", EventMessage::new("message", json!({})))).unwrap();
        assert!(observed(&log).is_empty());
    }

    #[test]
    fn a_panicking_sink_does_not_block_its_peers() {
        let bus = InProcessMessager::new();
        let log = new_log();
        bus.subscribe("alice", &[TeamTopic::Team], panicker())
            .unwrap();
        bus.subscribe("bob", &[TeamTopic::Team], recorder(&log, "bob"))
            .unwrap();

        block_on(bus.publish(TeamTopic::Team, EventMessage::new("team", json!({})))).unwrap();
        assert_eq!(observed_members(&log), ["bob"]);

        bus.subscribe("alice", &[TeamTopic::Message], panicker())
            .unwrap();
        block_on(bus.send("alice", EventMessage::new("message", json!({})))).unwrap();
        assert_eq!(observed_members(&log), ["bob"]);
    }

    #[test]
    fn start_and_stop_are_idempotent_and_keep_subscriptions() {
        let bus = InProcessMessager::new();
        let log = new_log();
        bus.subscribe("bob", &[TeamTopic::Team], recorder(&log, "bob"))
            .unwrap();
        assert!(bus.is_started());

        block_on(bus.stop()).unwrap();
        block_on(bus.stop()).unwrap();
        assert!(!bus.is_started());
        assert_eq!(bus.subscribers(TeamTopic::Team), ["bob"]);

        block_on(bus.publish(TeamTopic::Team, EventMessage::new("team", json!({})))).unwrap();
        block_on(bus.send("bob", EventMessage::new("message", json!({})))).unwrap();
        assert!(observed(&log).is_empty());

        block_on(bus.start()).unwrap();
        block_on(bus.start()).unwrap();
        assert!(bus.is_started());
        block_on(bus.publish(TeamTopic::Team, EventMessage::new("team", json!({})))).unwrap();
        block_on(bus.send("bob", EventMessage::new("message", json!({})))).unwrap();
        assert_eq!(observed_members(&log), ["bob", "bob"]);
    }

    #[test]
    fn subscribe_transport_records_sorted_unique_team_names() {
        let bus = InProcessMessager::new();
        assert!(bus.team_subscriptions().is_empty());
        for team in ["zeta", "alpha", "zeta"] {
            block_on(bus.subscribe_transport(team)).unwrap();
        }
        assert_eq!(bus.team_subscriptions(), ["alpha", "zeta"]);
    }

    #[test]
    fn registries_are_per_instance_not_process_global() {
        let first = InProcessMessager::new();
        let second = InProcessMessager::new();
        let log = new_log();
        first
            .subscribe("bob", &[TeamTopic::Team], recorder(&log, "bob"))
            .unwrap();
        assert!(second.subscribers(TeamTopic::Team).is_empty());
        block_on(second.publish(TeamTopic::Team, EventMessage::new("team", json!({})))).unwrap();
        block_on(second.send("bob", EventMessage::new("message", json!({})))).unwrap();
        assert!(observed(&log).is_empty());
    }

    #[test]
    fn from_config_uses_the_configured_sender() {
        let config = InProcessMessagerConfig {
            sender: "carol".to_owned(),
            peers: vec![MessagerPeerConfig {
                member_name: "bob".to_owned(),
                address: "tcp://127.0.0.1:5555".to_owned(),
            }],
        };
        let bus = InProcessMessager::from_config(&config);
        assert_eq!(bus.sender(), "carol");
        assert!(bus.is_started());
        assert_eq!(bus.name(), "inprocess");
        assert_eq!(InProcessMessager::default().sender(), "");
    }

    #[test]
    fn debug_reports_the_registry_shape_without_leaking_handlers() {
        let bus = InProcessMessager::with_sender("leader");
        bus.subscribe("bob", &[TeamTopic::Team], recorder(&new_log(), "bob"))
            .unwrap();
        let rendered = format!("{bus:?}");
        assert!(rendered.contains("InProcessMessager"));
        assert!(rendered.contains("sender: \"leader\""));
        assert!(rendered.contains("started: true"));
        assert!(rendered.contains("topics: 1"));
    }

    #[test]
    fn topic_wire_strings_match_the_python_grammar() {
        assert_eq!(TeamTopic::Team.as_str(), "team");
        assert_eq!(TeamTopic::Task.as_str(), "task");
        assert_eq!(TeamTopic::Message.as_str(), "message");
        for category in [TeamTopic::Team, TeamTopic::Task, TeamTopic::Message] {
            let topic = category.topic("sess-1", "team-1");
            assert_eq!(
                TeamTopic::parse(&topic),
                Some(("sess-1".to_owned(), "team-1".to_owned(), category))
            );
        }
        assert_eq!(
            TeamTopic::Message.topic("a:b", "c%d"),
            "session:a%3Ab:team:c%25d:message"
        );
    }
}
