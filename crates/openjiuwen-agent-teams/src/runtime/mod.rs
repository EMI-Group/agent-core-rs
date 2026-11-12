//! The team runtime: `TeamAgent` (leader and teammate), the leader's coordination loop, the
//! organisation backend, the runtime manager/pool and the model-pool allocator.
//!
//! Python parity sources: `openjiuwen/agent_teams/agent/team_agent.py`,
//! `agent/agent_configurator.py`, `runtime/{manager,dispatch,pool,gate}.py` and
//! `models/allocator.py`.
//!
//! The module owns orchestration only: rows live in [`crate::storage`], task decisions in
//! [`crate::taskboard`], member status in [`crate::member`], transports in [`crate::messager`].
//!
//! | File | Contents |
//! | --- | --- |
//! | [`deps`] | [`TeamBuildDeps`]: the injected build inputs and their defaults |
//! | [`allocator`] | [`ModelAllocator`]: round-robin / by-name model-pool selection |
//! | [`dispatch`] | [`decide_run_action`]: whether a run creates, attaches, resumes or is rejected |
//! | [`initial_input`] | [`parse_initial_input`]: caller request → leader query + deliveries |
//! | [`backend`] | [`TeamBackend`]: members, mail, events, task board and snapshot facade |
//! | [`build`] | the blueprint → runtime build path (`SpecBuild::build` / `build_with`) |
//! | [`spawner`] | [`TeamMemberSpawner`]: the harness-backed `MemberSpawner` |
//! | [`team_agent`] | [`TeamAgent`] and [`SpecBuild`] |
//! | [`leader`] | the coordination loop (`select!` over chunks, events, cancellation, settle tick) |
//! | [`pool`] | [`TeamRuntimePool`]: session → team runtime map |
//! | [`manager`] | [`TeamRuntimeManager`] and the process-global registry |
//! | [`adapter`] | the `openjiuwen_agent::Agent` and `openjiuwen_multiagent::Team` seams |
//! | `tests` | the crossed-seam integration suite |
/// The `openjiuwen_agent::Agent` and `openjiuwen_multiagent::Team` seams over [`team_agent::TeamAgent`].
pub mod adapter;
/// Model-pool selection: round-robin or by model name.
pub mod allocator;
/// The organisation facade a running team exposes: members, mail, events, tasks and snapshots.
pub mod backend;
/// The blueprint → runtime build path (`TeamAgentSpec::build` / `build_with`).
pub mod build;
/// Build dependencies and their defaults.
pub mod deps;
/// Run admission: create, attach, resume or reject.
pub mod dispatch;
/// Caller request parsing into a leader query and pre-routed deliveries.
pub mod initial_input;
/// The leader's coordination loop.
pub mod leader;
/// Process-wide and local runtime management.
pub mod manager;
/// The session → team runtime pool.
pub mod pool;
/// The harness-backed member spawner.
pub mod spawner;
/// The team agent: one type for the leader and every teammate.
pub mod team_agent;

#[cfg(test)]
mod tests;

pub use allocator::{ModelAllocator, TeamModelAllocation};
pub use backend::TeamBackend;
pub use deps::TeamBuildDeps;
pub use dispatch::{decide_run_action, LiveTeamState, RunAction, RunActionKind};
pub use initial_input::{parse_initial_input, InitialInput};
pub use leader::SETTLE_POLL_INTERVAL;
pub use manager::{
    clear_global_registry, global_registry, set_global_registry, try_global_registry,
    TeamRuntimeManager,
};
pub use pool::TeamRuntimePool;
pub use spawner::TeamMemberSpawner;
pub use team_agent::{SpecBuild, TeamAgent};
