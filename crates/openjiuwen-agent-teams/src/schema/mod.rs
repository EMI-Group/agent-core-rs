//! The crate's data model: specs, statuses, task records, events and stream frames.
//!
//! Everything here is pure data plus pure helpers — no I/O, no clocks, no global state — because
//! every other module of the crate codes against these shapes. The node contract lives in
//! `src/schema/CONTEXT.md`; read it before changing a field, a default or a wire string.
//!
//! Module map:
//!
//! | Module | Types |
//! | --- | --- |
//! | [`deep_agent`] | [`DeepAgentSpec`], [`ModelRef`], [`ToolSpec`], [`RailSpec`], [`McpServerSpec`], [`SubAgentSpec`], [`WorkspaceSpec`] |
//! | [`events`] | [`EventMessage`], [`TeamTopic`], [`EventPayload`] and every payload struct |
//! | [`member_spec`] | [`LeaderSpec`], [`MemberSpecBase`], [`TeamMemberSpec`], [`PredefinedMemberSpec`], [`TinyAgentSpec`], [`ModelPoolEntry`] |
//! | [`spec`] | [`TeamAgentSpec`], [`LeaderSpec`], member specs, transport/storage/workspace specs and spec enums |
//! | [`status`] | [`MemberStatus`], [`ExecutionStatus`], [`TaskStatus`], transition tables |
//! | [`stream`] | [`TeamOutputSchema`] and the stream frame markers |
//! | [`task`] | [`TaskRecord`], [`TaskSummary`], [`TaskDetail`], [`TaskGraphSpec`], [`ReviewVote`], [`TaskPatch`] |
//! | [`team`] | [`TeamRole`], [`TeamSpec`], [`TeamRuntimeContext`], [`TeamSnapshot`] |

pub mod deep_agent;
pub mod events;
pub mod member_spec;
pub mod spec;
pub mod status;
pub mod stream;
pub mod task;
pub mod team;

// Flat re-export of every public item; the per-module lists live in each file's own docs.
pub use deep_agent::*;
pub use events::*;
pub use member_spec::*;
pub use spec::*;
pub use status::*;
pub use stream::*;
pub use task::*;
pub use team::*;

pub use openjiuwen_harness_protocol::HarnessState;
