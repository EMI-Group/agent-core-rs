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
//! - the agents-as-tools strategy (a later addition) wires the sub-agent tools
//!   directly onto the root agent and invokes it in-process, bypassing the bus.
//!
//! Every strategy is a [`TeamStrategy`](crate::TeamStrategy) object attached to
//! the crate's single team type ([`BaseTeam`](crate::BaseTeam)): the team owns
//! the card, configuration, member registry, and runtime, while the strategy
//! decides how one invocation becomes member calls. A team is built with
//! [`hierarchical_msgbus_team`] plus the supervisor factory
//! [`create_supervisor`], which returns the `(card, provider)` pair the team
//! registers like any other member.

mod msgbus;
mod supervisor;

pub use msgbus::{hierarchical_msgbus_team, HierarchicalMsgbusConfig, HierarchicalMsgbusStrategy};
pub use supervisor::{create_supervisor, SupervisorAgent, SupervisorConfig};
