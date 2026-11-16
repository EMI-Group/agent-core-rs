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
//! | [`routing`] | [`TeamModelRouter`]: pool + router selection and the effective member spec |
//! | [`spec_meta`] | the metadata keys the build path reads (spawn, workspace, org, attach, KV, checkpoint) |
//! | `checkpoints` | per-member checkpoint identity and store resolution (leader and teammates) |
//! | `memory_bind` | team-memory resolution and the lifecycle hooks (session, join/leave, flush/export) |
//! | `observer` | the member observer feeding status, rounds, chunks and signals |
//! | [`observability_bind`] | [`TeamSignals`]: observability hooks + monitor fan-out |
//! | [`org_bind`] | [`OrgBinding`]: chart, delegation router, org task pool and mailbox |
//! | [`attach_host`] | [`AttachHost`] / [`BackendAttachHandler`]: Unix-socket attach hosting |
//! | [`dispatch`] | [`decide_run_action`]: whether a run creates, attaches, resumes or is rejected |
//! | [`health`] | [`TeamHealth`] / [`MemberHealth`]: the member-liveness projection of a team |
//! | [`initial_input`] | [`parse_initial_input`]: caller request → leader query + deliveries |
//! | [`backend`] | [`TeamBackend`]: members, mail, events, task board and snapshot facade |
//! | `tool_dispatch` | the four member-facing team tools answered against a [`TeamBackend`] |
//! | `member_roles` | role capabilities: which roles own a local harness and which cannot spawn |
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
/// Attach hosting: bind, serve and shut down the Unix-socket server.
pub mod attach_host;
/// The organisation facade a running team exposes: members, mail, events, tasks and snapshots.
pub mod backend;
/// The blueprint → runtime build path (`TeamAgentSpec::build` / `build_with`).
pub mod build;
/// Per-member checkpoint identity and store resolution shared by the build path and the spawner.
pub(crate) mod checkpoints;
/// Build dependencies and their defaults.
pub mod deps;
/// Run admission: create, attach, resume or reject.
pub mod dispatch;
/// The member-liveness projection of a running team.
pub mod health;
/// Caller request parsing into a leader query and pre-routed deliveries.
pub mod initial_input;
/// The leader's coordination loop.
pub mod leader;
/// Process-wide and local runtime management.
pub mod manager;
/// Role capabilities: which roles own a local harness and which cannot be spawned.
pub(crate) mod member_roles;
/// Team-memory resolution and the lifecycle hooks the runtime drives.
pub(crate) mod memory_bind;
/// Observability hooks and the optional monitor sink, fanned out from one place.
pub mod observability_bind;
/// The member observer that drives status, rounds, chunks and signals.
pub(crate) mod observer;
/// The organisation layer bound to a running team.
pub mod org_bind;
/// The session → team runtime pool.
pub mod pool;
/// Model pool + router selection per member.
pub mod routing;
/// The harness-backed member spawner.
pub mod spawner;
/// The team agent: one type for the leader and every teammate.
pub mod team_agent;
/// The in-process team-tool dispatcher shared by delivery and attach hosting.
pub(crate) mod tool_dispatch;

/// Spec-metadata keys and their parsers.
pub mod spec_meta;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod wiring_tests;

pub use allocator::{ModelAllocator, TeamModelAllocation};
pub use attach_host::{AttachHost, BackendAttachHandler};
pub use backend::TeamBackend;
pub use deps::TeamBuildDeps;
pub use dispatch::{decide_run_action, LiveTeamState, RunAction, RunActionKind};
pub use health::{MemberHealth, TeamHealth};
pub use initial_input::{parse_initial_input, InitialInput};
pub use leader::SETTLE_POLL_INTERVAL;
pub use manager::{
    clear_global_registry, global_registry, set_global_registry, try_global_registry,
    TeamRuntimeManager,
};
pub use observability_bind::TeamSignals;
pub use org_bind::OrgBinding;
pub use pool::TeamRuntimePool;
pub use routing::{ModelSelection, TeamModelRouter};
pub use spawner::TeamMemberSpawner;
pub use spec_meta::{
    checkpoint_config, AttachConfig, CheckpointConfig, CheckpointStoreConfig, ForkDirective,
    OrganizationBlock,
};
pub use team_agent::{SpecBuild, TeamAgent};
