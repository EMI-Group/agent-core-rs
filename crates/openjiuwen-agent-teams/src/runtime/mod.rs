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
//! | [`spawner`] | [`TeamMemberSpawner`]: the harness-backed `MemberSpawner` |
//! | [`team_agent`] | [`TeamAgent`], [`SpecBuild`] and the build path |
//! | [`leader`] | the coordination loop (`select!` over chunks, events, cancellation, settle tick) |
//! | [`pool`] | [`TeamRuntimePool`]: session → team runtime map |
//! | [`manager`] | [`TeamRuntimeManager`] and the process-global registry |
//! | [`adapter`] | the `openjiuwen_agent::Agent` and `openjiuwen_multiagent::Team` seams |
/// Injected build inputs of one team runtime.
pub mod allocator;
/// The `openjiuwen_agent::Agent` and `openjiuwen_multiagent::Team` seams over [`team_agent::TeamAgent`].
pub mod adapter;
/// The organisation facade a running team exposes: members, mail, events, tasks and snapshots.
pub mod backend;
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
