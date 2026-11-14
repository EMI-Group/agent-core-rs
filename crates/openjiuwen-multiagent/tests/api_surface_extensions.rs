#![allow(dead_code)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Compile-time pinning of the additive Rust extensions' public surface.
//!
//! This suite is the contract test for the modules that have no Python
//! counterpart (`routing`, `inbox`, `communicable`) plus the Python-parity
//! introspection methods they exposed on `TeamRuntime` and `MessageBus`. It
//! fails to compile when a crate-root re-export, an inherent constructor, a
//! free-function signature, a field, or a serde/schemars implementation drifts
//! from `src/CONTEXT.md`; behaviour itself lives in the module unit tests and
//! the focused `routing`/`inbox`/`communicable` suites.
//!
//! Everything here is in-process: no model, no network, no timers.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_multiagent::{
    broadcast_targets, CommunicableAgent, GroupBroadcast, GroupDelivery, GroupRegistry,
    InboxFilter, InboxFilterRegistry, MessageBus, MessageBusConfig, MessageEnvelope,
    MessageHandler, RoutingTable, RuntimeConfig, TeamRuntime, GROUP_METADATA_KEY,
};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::{json, Value};

/// A [`MessageHandler`] that exists to prove the trait is object-safe and to
/// build a bus without a delivery path.
struct NoopHandler;

#[async_trait]
impl MessageHandler for NoopHandler {
    async fn handle_p2p(&self, _envelope: &MessageEnvelope) -> Result<Value> {
        Ok(Value::Null)
    }

    async fn handle_pubsub(&self, _envelope: &MessageEnvelope, _subscriber: &str) -> Result<()> {
        Ok(())
    }
}

/// Asserts `T` is `Send + Sync`.
fn assert_send_sync<T: Send + Sync>() {}

/// Asserts a public data type round-trips through JSON unchanged.
fn round_trip<T>(value: &T)
where
    T: Serialize + DeserializeOwned + PartialEq + std::fmt::Debug,
{
    let encoded = serde_json::to_value(value).expect("public data types serialize");
    let decoded: T = serde_json::from_value(encoded).expect("public data types deserialize");
    assert_eq!(&decoded, value);
}

/// Returns the sorted root property names of a JSON Schema.
fn schema_properties(schema: &Value) -> Vec<String> {
    let properties = schema
        .get("properties")
        .and_then(Value::as_object)
        .expect("a struct schema is an object with properties");
    let mut names: Vec<String> = properties.keys().cloned().collect();
    names.sort_unstable();
    names
}

/// References the runtime's Python-parity introspection methods and the group
/// broadcast, so a removed or renamed method fails to compile.
fn pin_runtime_introspection(runtime: &TeamRuntime) {
    let _: bool = runtime.is_running();
    let _: BTreeMap<String, Vec<String>> = runtime.list_subscriptions();
    let _: Vec<String> = runtime.list_agent_subscriptions("member");
    let _: usize = runtime.get_subscription_count();
    let table: RoutingTable = runtime.routing_table();
    assert_eq!(table.team_id, runtime.team_id());
    let _ = TeamRuntime::broadcast_to_group;
}

/// References the bus's Python-parity introspection methods, so a removed or
/// renamed method fails to compile.
fn pin_bus_introspection(bus: &MessageBus) {
    let _: usize = bus.remove_all_subscriptions("member");
    let _: BTreeMap<String, Vec<String>> = bus.list_subscriptions();
    let _: Vec<String> = bus.list_agent_subscriptions("member");
    let _: usize = bus.get_subscription_count();
}

#[test]
fn rust_routing_extensions_are_re_exported_at_the_root() {
    assert_eq!(GROUP_METADATA_KEY, "group");

    let groups = GroupRegistry::new();
    assert_eq!(groups.group_count(), 0);
    assert_eq!(
        groups
            .register_group("eng", ["alice".to_owned(), "bob".to_owned()])
            .unwrap(),
        2
    );
    assert!(groups.add_member("eng", "carol").unwrap());
    assert!(!groups.add_member("eng", "carol").unwrap());
    assert_eq!(
        groups.group_members("eng"),
        vec!["alice".to_owned(), "bob".to_owned(), "carol".to_owned()]
    );
    assert_eq!(groups.groups(), vec!["eng".to_owned()]);
    assert_eq!(groups.groups_of("alice"), vec!["eng".to_owned()]);
    assert_eq!(groups.member_count(), 3);
    assert!(groups.remove_member("eng", "carol"));
    assert!(!groups.remove_member("eng", "carol"));
    assert!(groups.unregister_group("eng"));
    assert!(!groups.unregister_group("eng"));
    assert_eq!(groups.clear(), 0);

    // The sender is excluded by default, sorted, and deduplicated.
    let _: fn(Vec<String>, Option<&str>, bool) -> Vec<String> = broadcast_targets;
    assert_eq!(
        broadcast_targets(
            ["bob".to_owned(), "alice".to_owned(), "bob".to_owned()],
            Some("bob"),
            false
        ),
        vec!["alice".to_owned()]
    );
    assert_eq!(
        broadcast_targets(["bob".to_owned(), "alice".to_owned()], Some("bob"), true),
        vec!["alice".to_owned(), "bob".to_owned()]
    );

    // The broadcast outcome keeps its documented fields and helpers.
    let broadcast = GroupBroadcast {
        group: "eng".to_owned(),
        recipients: vec!["alice".to_owned(), "bob".to_owned()],
        skipped: vec!["zoe".to_owned()],
        deliveries: vec![
            GroupDelivery {
                agent_id: "alice".to_owned(),
                outcome: Ok(json!({"ok": true})),
            },
            GroupDelivery {
                agent_id: "bob".to_owned(),
                outcome: Err(Error::validation("boom")),
            },
        ],
    };
    assert_eq!(broadcast.succeeded(), 1);
    assert_eq!(broadcast.failed(), 1);
    assert_eq!(broadcast.value_of("alice"), Some(&json!({"ok": true})));
    assert!(broadcast.value_of("bob").is_none());
    let results = broadcast.into_results();
    assert_eq!(results.len(), 2);
    assert!(results["alice"].is_ok());
    assert!(results["bob"].is_err());

    // The snapshot resolves a group first, then topic patterns.
    let table = RoutingTable {
        team_id: "crew".to_owned(),
        groups: BTreeMap::from([("eng".to_owned(), vec!["alice".to_owned()])]),
        topics: BTreeMap::from([("news.*".to_owned(), vec!["bob".to_owned()])]),
        agents: BTreeMap::from([(
            "alice".to_owned(),
            vec!["eng".to_owned(), "news.*".to_owned()],
        )]),
    };
    assert_eq!(table.route("eng"), vec!["alice".to_owned()]);
    assert_eq!(table.route("news.tech"), vec!["bob".to_owned()]);
    assert_eq!(table.route("unknown"), Vec::<String>::new());
    assert_eq!(
        table.destinations("alice"),
        vec!["eng".to_owned(), "news.*".to_owned()]
    );

    assert_send_sync::<GroupRegistry>();
    assert_send_sync::<RoutingTable>();
}

#[test]
fn rust_inbox_extensions_are_re_exported_at_the_root() {
    let filter = InboxFilter::new()
        .allow_senders(["alice"])
        .deny_senders(["mallory"])
        .allow_topics(["news.*"])
        .require_metadata(GROUP_METADATA_KEY, "eng")
        .require_session();
    let accepted = MessageEnvelope::pubsub(json!({"n": 1}), "alice", "news.tech")
        .with_session("s-1")
        .with_metadata(GROUP_METADATA_KEY, json!("eng"));
    assert!(filter.accepts(&accepted));
    let denied = MessageEnvelope::pubsub(json!({"n": 1}), "mallory", "news.tech")
        .with_session("s-1")
        .with_metadata(GROUP_METADATA_KEY, json!("eng"));
    assert!(!filter.accepts(&denied));

    let registry = InboxFilterRegistry::new();
    assert!(registry.is_empty());
    assert_eq!(registry.len(), 0);
    // An agent without a filter accepts everything.
    assert!(registry.accepts("bob", &denied));
    registry.set("bob", InboxFilter::new().deny_senders(["mallory"]));
    assert_eq!(registry.agents(), vec!["bob".to_owned()]);
    assert_eq!(registry.len(), 1);
    assert!(!registry.is_empty());
    assert!(registry.get("bob").is_some());
    assert!(registry.accepts("bob", &accepted));
    assert!(!registry.accepts("bob", &denied));
    assert!(registry.remove("bob"));
    assert!(registry.accepts("bob", &denied));
    registry.clear();

    assert_send_sync::<InboxFilterRegistry>();
    assert_send_sync::<CommunicableAgent>();
}

#[test]
fn runtime_and_bus_introspection_methods_are_reachable() {
    let runtime = TeamRuntime::new(RuntimeConfig::new().with_team_id("api-surface"));
    pin_runtime_introspection(&runtime);

    let bus = MessageBus::new(
        MessageBusConfig::new().with_team_id("api-surface"),
        Arc::new(NoopHandler),
    );
    pin_bus_introspection(&bus);
}

#[tokio::test]
async fn communicable_agent_is_re_exported_at_the_root() {
    let _: fn(String) -> CommunicableAgent = CommunicableAgent::new;
    let agent = CommunicableAgent::new("alice");
    assert!(!agent.is_bound());
    assert!(agent.runtime().is_none());
    assert_eq!(agent.agent_id().as_deref(), Some("alice"));

    // Every messaging call fails with the Python-parity validation error while
    // unbound.
    let errors = [
        agent.send(json!({}), "bob", None).await.unwrap_err(),
        agent.publish(json!({}), "news", None).await.unwrap_err(),
        agent.subscribe("news").await.unwrap_err(),
        agent.unsubscribe("news").await.unwrap_err(),
    ];
    for error in errors {
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(
            error.message(),
            "agent 'alice' is not bound to a team runtime"
        );
    }

    let runtime = TeamRuntime::new(RuntimeConfig::new().with_team_id("api-surface"));
    agent.bind_runtime(Arc::clone(&runtime), "alice");
    assert!(agent.is_bound());
    assert!(agent.runtime().is_some());
    // The same `(runtime, agent_id)` pair is a silent no-op.
    agent.bind_runtime(Arc::clone(&runtime), "alice");
    assert!(agent.is_bound());

    agent.unbind();
    assert!(!agent.is_bound());
    assert_eq!(agent.agent_id().as_deref(), Some("alice"));
    let error = agent.send(json!({}), "bob", None).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[test]
fn extension_data_models_round_trip_and_keep_their_schema_names() {
    round_trip(&InboxFilter::new());
    round_trip(
        &InboxFilter::new()
            .allow_senders(["alice"])
            .deny_senders(["mallory"])
            .allow_topics(["news.*"])
            .require_metadata("group", "eng")
            .require_session(),
    );
    round_trip(&RoutingTable::default());
    round_trip(&RoutingTable {
        team_id: "crew".to_owned(),
        groups: BTreeMap::from([("eng".to_owned(), vec!["alice".to_owned()])]),
        topics: BTreeMap::from([("news.*".to_owned(), vec!["alice".to_owned()])]),
        agents: BTreeMap::from([("alice".to_owned(), vec!["eng".to_owned()])]),
    });

    let filter =
        serde_json::to_value(schemars::schema_for!(InboxFilter)).expect("schema serializes");
    assert_eq!(
        schema_properties(&filter),
        [
            "allow_senders",
            "deny_senders",
            "metadata",
            "require_session",
            "topics"
        ]
    );

    let table =
        serde_json::to_value(schemars::schema_for!(RoutingTable)).expect("schema serializes");
    assert_eq!(
        schema_properties(&table),
        ["agents", "groups", "team_id", "topics"]
    );
}
