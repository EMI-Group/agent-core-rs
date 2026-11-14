//! Teams of agents: team cards and configuration, an in-process message bus, a
//! team runtime, and the `TeamStrategy` seam (handoff and hierarchical
//! strategies build on it).
//!
//! Maps to the Python package `core.multi_agent` in the openJiuwen reference
//! implementation: a `Team` owns a card, a [`TeamRuntime`], a [`MessageBus`]
//! transport and a strategy object that decides how members collaborate.
#![forbid(unsafe_code)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]
#![allow(clippy::module_name_repetitions)]

mod agent_tool;
mod card;
mod config;
mod envelope;
mod error;
mod handoff;
pub mod hierarchical;
mod member;
mod message_bus;
mod runtime;
mod session;
mod strategy;
mod subscription;
mod team;

pub mod ability;
pub mod autosubscribe;
pub mod communicable;
pub mod events;
pub mod member_checkpoint;
pub mod pyrepr;
pub mod resources;
pub mod transport;

pub use agent_tool::AgentTool;
pub use card::{EventDrivenTeamCard, TeamCard};
pub use config::TeamConfig;
pub use envelope::MessageEnvelope;
pub use error::TeamError;
pub use handoff::{
    extract_handoff_signal, extract_handoff_signal_from_observations,
    extract_handoff_signal_from_session, extract_interrupt_signal, flush_team_session,
    handoff_team, ContainerAgent, HandoffConfig, HandoffHistoryEntry, HandoffOrchestrator,
    HandoffRegistry, HandoffRequest, HandoffRoute, HandoffSignal, HandoffSnapshot, HandoffStrategy,
    HandoffTeamConfig, HandoffTool, TeamInterruptSignal, TerminationCondition,
    CONTEXT_HISTORY_STATE_KEY, COORDINATOR_STATE_KEY, HANDOFF_MESSAGE_KEY, HANDOFF_REASON_KEY,
    HANDOFF_TARGET_KEY, HISTORY_STATE_KEY,
};
pub use hierarchical::{
    create_supervisor, hierarchical_msgbus_team, hierarchical_tools_team, HierarchicalMsgbusConfig,
    HierarchicalMsgbusStrategy, HierarchicalToolsConfig, HierarchicalToolsStrategy,
    SupervisorAgent, SupervisorConfig,
};
pub use member::{AgentProvider, BindHook, TeamAgent, ToolHost};
pub use message_bus::{MessageBus, MessageBusConfig, MessageHandler};
pub use runtime::{RuntimeConfig, TeamRuntime};
pub use session::{make_team_session, team_stream, SessionLease};
pub use strategy::{MemberOptions, StrategyContext, TeamStrategy};
pub use subscription::{topic_matches, SubscriptionManager};
pub use team::{BaseTeam, Team};

// --- extended subsystems -----------------------------------------------------
//
// The modules below used to be reachable only through their module path; the
// re-exports publish them at the crate root, so a consumer needs one path for
// the whole team surface. Nothing is renamed: every item keeps the name it has
// in its module.

/// Ability routing for a supervisor's model tool calls (Python
/// `teams/hierarchical_msgbus/p2p_ability_manager.py`).
pub use ability::{AbilityCall, AbilityExecutor, AbilityResult, AbilityRoute, P2PAbilityManager};
/// Card-driven auto-subscription (Python `AgentTeamCard.subscriptions`).
pub use autosubscribe::{
    apply, apply_agent, derive, derive_from_event_card, derive_with_overrides, derived_for_member,
    AutoSubscriptionMode, AutoSubscriptionPlan, SUBSCRIBE_TAG_PREFIX,
};
/// One agent id bound to a team runtime, with P2P and pub/sub helpers (Python
/// `team_runtime/communicable_agent.py`).
pub use communicable::CommunicableAgent;
/// Typed team events (the Python `AgentTeamEvents` notifications).
pub use events::{
    TeamEvent, TeamEventBus, TeamEventFilter, TeamEventKind, TeamEventListener,
    AGENT_P2P_RECEIVED_EVENT, AGENT_PUBSUB_RECEIVED_EVENT, TEAM_EVENT_SCOPE,
};
/// Per-member checkpoint identity over one shared checkpointer.
pub use member_checkpoint::{
    member_checkpoint_id, member_checkpoint_key, MemberCheckpointer, MEMBER_CHECKPOINT_NAMESPACE,
};
/// Python literal parsing, the `ast.literal_eval` fallback of signal extraction.
pub use pyrepr::{literal_or_json, parse_python_literal, render_python_value, MAX_LITERAL_DEPTH};
/// The shared resource manager of a team (Python `Runner.resource_mgr` shape).
pub use resources::{LocalResourceMgr, TeamResourceMgr};
/// The transport seam a team's messages travel over.
pub use transport::{
    build_transport, InProcessTransport, RedisTransport, RedisTransportConfig, TeamTransport,
    TransportConfig, TransportKind,
};
