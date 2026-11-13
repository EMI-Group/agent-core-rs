//! Hierarchical teams: a supervisor drives a set of sub-agents.
//!
//! Python parity: `openjiuwen/core/multi_agent/teams/hierarchical_msgbus/` and
//! `.../hierarchical_tools/`. Both topologies put one supervisor (a `ReActAgent`
//! whose model can call every sub-agent) above a flat member list; they differ
//! only in how an input reaches that supervisor:
//!
//! - [`HierarchicalMsgbusStrategy`] dispatches the input with a single P2P
//!   [`send`](crate::TeamRuntime::send) to the supervisor registered in the team
//!   runtime, and the supervisor reaches its sub-agents through the same bus
//!   (they are registered as members and exposed to its model as tools);
//! - [`HierarchicalToolsStrategy`] wires the sub-agent tools directly onto the
//!   root agent — one [`AgentTool`](crate::AgentTool) per member added with the
//!   root as its `parent_agent_id` — pins that instance so later sends reach it,
//!   and streams the root in-process, bypassing the bus.
//!
//! Every strategy is a [`TeamStrategy`](crate::TeamStrategy) object attached to
//! the crate's single team type ([`BaseTeam`](crate::BaseTeam)): the team owns
//! the card, configuration, member registry, and runtime, while the strategy
//! decides how one invocation becomes member calls. A team is built with
//! [`hierarchical_msgbus_team`] plus the supervisor factory
//! [`create_supervisor`], which returns the `(card, provider)` pair the team
//! registers like any other member, or with [`hierarchical_tools_team`], whose
//! root agent is supplied as a card and registered by the caller.
//!
//! The supervisor's ability routing ([`P2PAbilityManager`]) is re-exported here
//! as well as from the crate root, because Python keeps it at
//! `openjiuwen/core/multi_agent/teams/hierarchical_msgbus/p2p_ability_manager.py`.

mod msgbus;
mod supervisor;
mod tools;

pub use msgbus::{hierarchical_msgbus_team, HierarchicalMsgbusConfig, HierarchicalMsgbusStrategy};
pub use supervisor::{create_supervisor, SupervisorAgent, SupervisorConfig};
pub use tools::{hierarchical_tools_team, HierarchicalToolsConfig, HierarchicalToolsStrategy};

// Python parity: `P2PAbilityManager` lives in
// `openjiuwen/core/multi_agent/teams/hierarchical_msgbus/p2p_ability_manager.py`,
// so it is reachable both as `crate::P2PAbilityManager` and under this module
// path.
pub use crate::ability::{AbilityCall, AbilityResult, AbilityRoute, P2PAbilityManager};
