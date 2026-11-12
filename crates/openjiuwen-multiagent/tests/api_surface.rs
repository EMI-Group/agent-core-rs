#![allow(dead_code)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Compile-time pinning of the crate's public surface.
//!
//! This suite is a contract test: it fails to compile when a crate-root
//! re-export, an inherent constructor, a free-function signature, a trait
//! object, or a serde/schemars implementation drifts from `CONTEXT.md`. It
//! asserts as little behaviour as possible — the authoritative behaviour lives
//! in the module unit tests and the focused integration suites — and instead
//! references every documented item so a rename or signature change is caught
//! at build time rather than at a reviewer's discretion.
//!
//! Everything here is in-process: no model, no network, no timers.

use std::collections::BTreeMap;
use std::sync::{Arc, Weak};

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::{FutureExt, StreamExt};
use openjiuwen_agent::{Agent, AgentCard, ReActAgent};
use openjiuwen_common::card::Card;
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::tool::Tool;
use openjiuwen_multiagent::{
    create_supervisor, hierarchical_msgbus_team, hierarchical_tools_team, make_team_session,
    team_stream, topic_matches, AgentProvider, AgentTool, BaseTeam, BindHook, EventDrivenTeamCard,
    HierarchicalMsgbusConfig, HierarchicalMsgbusStrategy, HierarchicalToolsConfig,
    HierarchicalToolsStrategy, MemberOptions, MessageBus, MessageBusConfig, MessageEnvelope,
    MessageHandler, RuntimeConfig, SessionLease, StrategyContext, SubscriptionManager,
    SupervisorAgent, SupervisorConfig, Team, TeamAgent, TeamCard, TeamConfig, TeamError,
    TeamRuntime, TeamStrategy, ToolHost,
};
use openjiuwen_session::{AgentTeamSession, Session, StreamItem};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::{json, Value};

/// A [`TeamStrategy`] that exists to prove the trait is object-safe.
struct PinStrategy;

#[async_trait]
impl TeamStrategy for PinStrategy {
    fn kind(&self) -> &'static str {
        "pin"
    }

    async fn invoke(
        &self,
        _ctx: &StrategyContext,
        input: Value,
        _session: Option<Arc<AgentTeamSession>>,
    ) -> Result<Value> {
        Ok(input)
    }

    fn stream(
        &self,
        _ctx: &StrategyContext,
        input: Value,
        _session: Option<Arc<AgentTeamSession>>,
    ) -> BoxStream<'static, Result<Value>> {
        futures::stream::once(async move { Ok(input) }).boxed()
    }
}

/// A [`MessageHandler`] that exists to prove the trait is object-safe.
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

/// Asserts `T` is `Clone`.
fn assert_clone<T: Clone>() {}

/// Asserts `T` is `PartialEq`.
fn assert_partial_eq<T: PartialEq>() {}

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

/// References [`team_stream`] so its generic signature is pinned at build time.
fn team_stream_reference(
    runtime: Arc<TeamRuntime>,
    card: TeamCard,
    message: Value,
    session: Option<Arc<AgentTeamSession>>,
) -> BoxStream<'static, Result<Value>> {
    team_stream(runtime, card, message, session, |_session| {
        async move { Ok(()) }.boxed()
    })
}

#[test]
fn crate_root_reexports_are_pinned_by_signature() {
    // Inherent constructors.
    let _: fn(TeamCard, TeamConfig) -> BaseTeam = BaseTeam::new;
    let _: fn(TeamCard, TeamConfig, Arc<dyn TeamStrategy>) -> BaseTeam = BaseTeam::with_strategy;
    let _: fn(TeamCard, TeamConfig, Arc<TeamRuntime>, Arc<dyn TeamStrategy>) -> BaseTeam =
        BaseTeam::with_runtime;
    let _: fn(MessageBusConfig, Arc<dyn MessageHandler>) -> MessageBus = MessageBus::new;
    let _: fn(RuntimeConfig) -> Arc<TeamRuntime> = TeamRuntime::new;
    let _: fn(Arc<dyn Agent>) -> TeamAgent = TeamAgent::new;
    let _: fn(Arc<ReActAgent>) -> TeamAgent = TeamAgent::react;
    let _: fn(Arc<dyn Agent>, Arc<dyn ToolHost>) -> TeamAgent = TeamAgent::with_tool_host;
    let _: fn(&AgentCard, Weak<TeamRuntime>) -> AgentTool = AgentTool::new;
    let _: fn(Value) -> MessageEnvelope = MessageEnvelope::new;
    let _: fn(Value, String, String) -> MessageEnvelope = MessageEnvelope::p2p;
    let _: fn(Value, String, String) -> MessageEnvelope = MessageEnvelope::pubsub;
    let _: fn(String) -> TeamCard = TeamCard::new;
    let _: fn() -> TeamConfig = TeamConfig::new;
    let _: fn() -> RuntimeConfig = RuntimeConfig::new;
    let _: fn() -> MessageBusConfig = MessageBusConfig::new;
    let _: fn() -> SubscriptionManager = SubscriptionManager::new;

    // Documented builders.
    let _: fn(TeamConfig, usize) -> TeamConfig = TeamConfig::configure_max_agents;
    let _: fn(TeamConfig, usize) -> TeamConfig = TeamConfig::configure_concurrency;
    let _: fn(TeamConfig, f64) -> TeamConfig = TeamConfig::configure_timeout;
    let _: fn(MessageBusConfig, usize) -> MessageBusConfig = MessageBusConfig::with_max_queue_size;
    let _: fn(MessageBusConfig, Option<f64>) -> MessageBusConfig =
        MessageBusConfig::with_process_timeout;
    let _: fn(RuntimeConfig, f64) -> RuntimeConfig = RuntimeConfig::with_p2p_timeout;

    // Free functions and the subscription registry.
    let _: fn(&TeamCard, &Value) -> Arc<AgentTeamSession> = make_team_session;
    let _: fn(&str, &str) -> bool = topic_matches;
    let _: fn(&SubscriptionManager, &str) -> Vec<String> = SubscriptionManager::subscribers;
    let _: fn(&SubscriptionManager, &str) -> Vec<String> = SubscriptionManager::topics_of;
    let _: fn(&SubscriptionManager) -> Vec<String> = SubscriptionManager::agents;
    let _: fn(&SubscriptionManager) -> usize = SubscriptionManager::subscription_count;
    let _: fn(&SubscriptionManager, &str) = SubscriptionManager::remove_agent;
    let _: fn(&SubscriptionManager, &str, &str) = SubscriptionManager::subscribe;
    let _: fn(&SubscriptionManager, &str, &str) = SubscriptionManager::unsubscribe;
    // The lease is async, so its signature is pinned by the calls below.
    let _ = SessionLease::acquire;
    let _ = team_stream_reference;

    // Trait-object coercions.
    let _: Arc<dyn Team> = Arc::new(BaseTeam::new(TeamCard::new("pin"), TeamConfig::default()));
    let _: Arc<dyn TeamStrategy> = Arc::new(PinStrategy);
    let _: Arc<dyn MessageHandler> = Arc::new(NoopHandler);
    let _: Arc<dyn ToolHost> = Arc::new(ReActAgent::new(AgentCard::new("host")));

    let runtime = TeamRuntime::new(RuntimeConfig::new());
    let _: Arc<dyn Tool> = Arc::new(AgentTool::new(
        &AgentCard::new("member"),
        Arc::downgrade(&runtime),
    ));

    // Member capabilities are plain aliases the runtime can hold.
    let provider: AgentProvider =
        Arc::new(|| TeamAgent::new(Arc::new(ReActAgent::new(AgentCard::new("member")))));
    assert_eq!(provider().agent().card().name, "member");

    let hook: BindHook = Arc::new(|_runtime: &Arc<TeamRuntime>, _agent_id: &str| {});
    hook(&runtime, "member");

    // The strategy context is a plain snapshot of the documented fields.
    let context = StrategyContext::new(
        TeamCard::new("pin"),
        TeamConfig::new(),
        vec![AgentCard::new("member")],
        Arc::clone(&runtime),
    );
    assert_eq!(context.card.name, "pin");
    assert_eq!(context.config.max_agents, 10);
    assert_eq!(context.members.len(), 1);
    assert_eq!(context.runtime.team_id(), "default");
}

// --- hierarchical strategies ---

#[test]
fn hierarchical_surface_is_pinned() {
    // Inherent constructors and free functions.
    let _: fn(AgentCard) -> HierarchicalMsgbusConfig = HierarchicalMsgbusConfig::new;
    let _: fn(AgentCard) -> HierarchicalToolsConfig = HierarchicalToolsConfig::new;
    let _: fn(HierarchicalMsgbusConfig) -> Arc<HierarchicalMsgbusStrategy> =
        HierarchicalMsgbusStrategy::new;
    let _: fn(HierarchicalToolsConfig) -> Arc<HierarchicalToolsStrategy> =
        HierarchicalToolsStrategy::new;
    let _: fn(AgentCard, Arc<dyn Model>, String) -> SupervisorConfig = SupervisorConfig::new;
    let _: fn(Vec<AgentCard>, SupervisorConfig) -> Result<(AgentCard, AgentProvider)> =
        SupervisorAgent::create;
    let _: fn(Vec<AgentCard>, SupervisorConfig) -> Result<(AgentCard, AgentProvider)> =
        create_supervisor;
    let _: fn(TeamCard, HierarchicalMsgbusConfig) -> BaseTeam = hierarchical_msgbus_team;
    let _: fn(TeamCard, HierarchicalToolsConfig) -> BaseTeam = hierarchical_tools_team;
    // `SupervisorAgent` is a unit-struct factory; its `Debug` shape is pinned too.
    let supervisor_agent: SupervisorAgent = SupervisorAgent;
    assert_eq!(format!("{supervisor_agent:?}"), "SupervisorAgent");

    // Both strategies coerce to the documented trait object and report their kind.
    let msgbus_config = HierarchicalMsgbusConfig::new(AgentCard::new("boss").with_id("boss"));
    let tools_config = HierarchicalToolsConfig::new(AgentCard::new("root").with_id("root"));
    let msgbus: Arc<dyn TeamStrategy> = HierarchicalMsgbusStrategy::new(msgbus_config.clone());
    let tools: Arc<dyn TeamStrategy> = HierarchicalToolsStrategy::new(tools_config.clone());
    assert_eq!(msgbus.kind(), "hierarchical_msgbus");
    assert_eq!(tools.kind(), "hierarchical_tools");

    // Each builder returns the single team type with the strategy attached.
    let msgbus_team = hierarchical_msgbus_team(TeamCard::new("pin-msg"), msgbus_config);
    assert_eq!(msgbus_team.kind(), "hierarchical_msgbus");
    assert_eq!(msgbus_team.team_id(), "pin-msg");
    let tools_team = hierarchical_tools_team(TeamCard::new("pin-tools"), tools_config);
    assert_eq!(tools_team.kind(), "hierarchical_tools");
    assert_eq!(tools_team.team_id(), "pin-tools");
}

#[test]
fn card_trait_and_builders_are_exposed() {
    let card = TeamCard::new("team")
        .with_id("card-id")
        .with_name("renamed")
        .with_description("desc")
        .with_agent_cards(vec![AgentCard::new("member").with_id("member")])
        .with_topic("topic")
        .with_version("2.0.0")
        .with_tags(vec!["tag".to_owned()]);

    assert_eq!(Card::id(&card), "card-id");
    assert_eq!(Card::name(&card), "renamed");
    assert_eq!(Card::description(&card), Some("desc"));
    assert_eq!(card.version, "2.0.0");
    assert_eq!(card.agent_cards.len(), 1);
    assert_eq!(card.tags, vec!["tag".to_owned()]);

    let event_driven = EventDrivenTeamCard {
        card,
        subscriptions: BTreeMap::from([("member".to_owned(), vec!["topic.*".to_owned()])]),
    };
    assert_eq!(event_driven.card.id, "card-id");
    assert_eq!(event_driven.subscriptions["member"], vec!["topic.*"]);

    assert_eq!(MemberOptions::default().parent_agent_id, None);
    assert_eq!(
        MemberOptions {
            parent_agent_id: Some("boss".to_owned()),
        }
        .parent_agent_id
        .as_deref(),
        Some("boss")
    );
}

#[test]
fn serde_round_trips_and_schemas_cover_public_data() {
    round_trip(&TeamCard::new("team").with_id("card-id"));
    round_trip(&EventDrivenTeamCard {
        card: TeamCard::new("team").with_id("card-id"),
        subscriptions: BTreeMap::from([("member".to_owned(), vec!["topic.*".to_owned()])]),
    });
    round_trip(&TeamConfig::new());
    round_trip(&MessageEnvelope::p2p(json!({"n": 1}), "a", "b").with_session("s-1"));
    round_trip(&MessageBusConfig::new());
    round_trip(&RuntimeConfig::new());
    round_trip(&MemberOptions {
        parent_agent_id: Some("boss".to_owned()),
    });
    // --- hierarchical strategies ---
    round_trip(&HierarchicalMsgbusConfig::new(
        AgentCard::new("boss").with_id("boss"),
    ));
    round_trip(&HierarchicalToolsConfig::new(
        AgentCard::new("root").with_id("root"),
    ));

    let card = serde_json::to_value(schemars::schema_for!(TeamCard)).expect("schema serializes");
    assert_eq!(
        schema_properties(&card),
        [
            "agent_cards",
            "description",
            "id",
            "name",
            "tags",
            "topic",
            "version"
        ]
    );

    let config =
        serde_json::to_value(schemars::schema_for!(TeamConfig)).expect("schema serializes");
    assert_eq!(
        schema_properties(&config),
        ["max_agents", "max_concurrent_messages", "message_timeout"]
    );

    let envelope =
        serde_json::to_value(schemars::schema_for!(MessageEnvelope)).expect("schema serializes");
    assert_eq!(
        schema_properties(&envelope),
        [
            "message",
            "message_id",
            "metadata",
            "recipient",
            "sender",
            "session_id",
            "topic_id"
        ]
    );

    let bus =
        serde_json::to_value(schemars::schema_for!(MessageBusConfig)).expect("schema serializes");
    assert_eq!(
        schema_properties(&bus),
        ["max_queue_size", "process_timeout", "team_id"]
    );

    let runtime =
        serde_json::to_value(schemars::schema_for!(RuntimeConfig)).expect("schema serializes");
    assert_eq!(
        schema_properties(&runtime),
        ["message_bus", "p2p_timeout", "team_id"]
    );

    let options =
        serde_json::to_value(schemars::schema_for!(MemberOptions)).expect("schema serializes");
    assert_eq!(schema_properties(&options), ["parent_agent_id"]);

    // Hierarchical configs keep their team limits flat and name the entry agent.
    let msgbus = serde_json::to_value(schemars::schema_for!(HierarchicalMsgbusConfig))
        .expect("schema serializes");
    assert_eq!(
        schema_properties(&msgbus),
        [
            "max_agents",
            "max_concurrent_messages",
            "message_timeout",
            "supervisor_agent",
            "timeout"
        ]
    );

    let tools = serde_json::to_value(schemars::schema_for!(HierarchicalToolsConfig))
        .expect("schema serializes");
    assert_eq!(
        schema_properties(&tools),
        [
            "max_agents",
            "max_concurrent_messages",
            "message_timeout",
            "root_agent"
        ]
    );

    // The flattened card keeps its team fields and adds the subscriptions map.
    let event_driven = serde_json::to_value(schemars::schema_for!(EventDrivenTeamCard))
        .expect("schema serializes");
    assert!(
        event_driven.is_object(),
        "the event-driven card exposes one object schema: {event_driven}"
    );
}

#[test]
fn team_error_converts_to_an_external_error() {
    let empty: Error = Error::external(TeamError::empty_response("a", "b"));
    assert_eq!(empty.kind(), ErrorKind::External);
    assert!(
        empty.message().contains("empty response from 'a' to 'b'"),
        "{empty}"
    );

    let converted: Error = TeamError::execution("boom").into();
    assert_eq!(converted.kind(), ErrorKind::External);
    assert!(
        converted.message().contains("team execution error: boom"),
        "{converted}"
    );
}

#[test]
fn marker_traits_are_implemented() {
    assert_clone::<TeamAgent>();
    assert_clone::<MessageEnvelope>();
    assert_partial_eq::<MessageEnvelope>();
    assert_clone::<TeamCard>();
    assert_partial_eq::<TeamCard>();
    assert_send_sync::<TeamCard>();
    assert_send_sync::<MessageEnvelope>();
    assert_send_sync::<TeamRuntime>();
    assert_send_sync::<BaseTeam>();
    assert_send_sync::<Arc<TeamRuntime>>();
}

#[tokio::test]
async fn team_trait_methods_are_reachable_through_the_object() {
    let team: Arc<dyn Team> = Arc::new(BaseTeam::new(
        TeamCard::new("pin").with_id("pin-id"),
        TeamConfig::default(),
    ));

    assert_eq!(team.card().name, "pin");
    assert_eq!(team.card().id, "pin-id");
    assert_eq!(team.config(), TeamConfig::default());
    assert_eq!(team.team_id(), "pin");
    assert_eq!(team.kind(), "none");
    assert_eq!(team.runtime().team_id(), "pin-id");
    assert_eq!(team.agent_count(), 0);
    assert_eq!(team.list_agents(), Vec::<String>::new());
    assert!(team.get_agent_card("missing").is_none());
    team.check_ready().unwrap();

    let provider: AgentProvider =
        Arc::new(|| TeamAgent::new(Arc::new(ReActAgent::new(AgentCard::new("member")))));
    team.add_agent(
        AgentCard::new("member").with_id("member"),
        Arc::clone(&provider),
    )
    .unwrap();
    team.add_member(
        AgentCard::new("other").with_id("other"),
        provider,
        MemberOptions::default(),
    )
    .unwrap();

    assert_eq!(team.agent_count(), 2);
    assert_eq!(
        team.list_agents(),
        vec!["member".to_owned(), "other".to_owned()]
    );
    assert_eq!(
        team.get_agent_card("member").map(|card| card.name),
        Some("member".to_owned())
    );

    team.subscribe("member", "topic.*").await.unwrap();
    team.unsubscribe("member", "topic.*").await.unwrap();

    team.start().await.unwrap();
    team.start().await.unwrap();
    team.stop().await.unwrap();
    team.stop().await.unwrap();

    // Membership validation is still the documented `Validation` error.
    let error = team
        .send(json!({}), "other", "ghost", None, None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    let error = team
        .publish(json!({}), "news", "ghost", None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);

    team.remove_agent("member").unwrap();
    team.remove_agent("other").unwrap();
    assert_eq!(team.agent_count(), 0);

    // A strategy-free team rejects execution with the documented error.
    let error = team.invoke(json!({}), None).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert_eq!(error.message(), "team has no strategy");

    let mut frames = team.stream(json!({}), None);
    let error = frames.next().await.unwrap().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert_eq!(error.message(), "team has no strategy");
    assert!(frames.next().await.is_none());
}

#[tokio::test]
async fn session_lease_and_stream_helpers_are_reachable() {
    let runtime = TeamRuntime::new(RuntimeConfig::new().with_team_id("api-surface"));
    let card = TeamCard::new("pin").with_id("pin-id");

    // A standalone lease owns and tears down the session it creates.
    let lease = SessionLease::acquire(
        &runtime,
        &card,
        &json!({"conversation_id": "lease-1"}),
        None,
    )
    .await;
    assert!(lease.is_standalone());
    assert_eq!(lease.session_id(), "lease-1");
    let _: &Arc<AgentTeamSession> = lease.session();
    lease.finish().await.unwrap();

    // A caller-provided session is borrowed, not owned.
    let caller = make_team_session(&card, &json!({"conversation_id": "lease-2"}));
    let borrowed =
        SessionLease::acquire(&runtime, &card, &json!({}), Some(Arc::clone(&caller))).await;
    assert!(!borrowed.is_standalone());
    assert_eq!(borrowed.session_id(), "lease-2");
    borrowed.finish().await.unwrap();

    // Frames written by a run reach the consumer as wire frames.
    let stream = team_stream(
        Arc::clone(&runtime),
        card,
        json!({"conversation_id": "stream-1"}),
        None,
        |session| {
            async move {
                session
                    .write_stream(StreamItem::output("message", None, json!({"text": "hi"})))
                    .await?;
                Ok(())
            }
            .boxed()
        },
    );
    let frames: Vec<Result<Value>> = stream.collect().await;
    assert_eq!(frames.len(), 1, "the run writes exactly one frame");
    assert_eq!(frames[0].as_ref().unwrap()["type"], "message");

    runtime.stop().await.unwrap();
}

#[tokio::test]
async fn bus_and_subscription_registry_are_reachable() {
    let bus = MessageBus::new(
        MessageBusConfig::new().with_team_id("api-surface"),
        Arc::new(NoopHandler),
    );
    assert!(!bus.is_started());
    assert_eq!(bus.team_id(), "api-surface");
    assert_eq!(bus.config().max_queue_size, 1000);

    bus.start().await.unwrap();
    assert!(bus.is_started());
    bus.add_subscription("agent", "news*");
    assert_eq!(bus.subscribers("news"), vec!["agent".to_owned()]);
    bus.remove_subscription("agent", "news*");
    assert_eq!(bus.subscribers("news"), Vec::<String>::new());
    bus.subscribe("topic").await.unwrap();
    bus.unsubscribe("topic").await.unwrap();
    bus.stop().await.unwrap();
    assert!(!bus.is_started());
}
