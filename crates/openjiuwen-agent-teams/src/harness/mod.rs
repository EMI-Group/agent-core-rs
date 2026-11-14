//! Adapter between a team member and the provider-neutral harness protocol.
//!
//! Python parity source: `openjiuwen/agent_teams/harness/` (`protocol_adapter.py`,
//! `native_harness.py`, `checkpoint.py`). The team runtime never talks to a harness implementation
//! directly: it builds one through [`MemberHarnessFactory`], wraps it in
//! [`HarnessMemberRuntime`], and observes it through
//! [`MemberObserver`](crate::member::MemberObserver).
//!
//! Module map:
//!
//! | File | Contents |
//! | --- | --- |
//! | [`adapter`] | [`MemberHarnessContext`], [`MemberHarnessFactory`], [`HarnessMemberRuntime`] |
//! | [`mapping`] | state/round/chunk projections onto the team vocabulary |
//! | [`checkpoint`] | [`NativeCheckpoint`], the portable member snapshot |
//! | [`checkpoint_store`] | per-member checkpoint persistence, policy and the protocol sink |
//! | [`native`] | [`NativeMemberFactory`], [`ModelResolver`], [`EnvModelResolver`] |
//! | `memory_tools` | the context's team-memory toolkit projection and its identity rule |
//!
//! The harness handle itself is `Arc<dyn openjiuwen_harness_protocol::HarnessProtocol>`; the
//! in-process implementation is `openjiuwen_harness::NativeHarness`, which already conforms to
//! that protocol. Distributed/external harness modes are not part of this module.
pub mod adapter;
pub mod checkpoint;
pub mod checkpoint_store;
pub mod mapping;
mod memory_tools;
pub mod native;
pub use adapter::{HarnessMemberRuntime, MemberHarnessContext, MemberHarnessFactory};
pub use checkpoint::NativeCheckpoint;
pub use checkpoint_store::{
    CheckpointResume, FileMemberCheckpointStore, InMemoryMemberCheckpointStore,
    MemberCheckpointPolicy, MemberCheckpointRecord, MemberCheckpointSink, MemberCheckpointStore,
};
pub use mapping::{
    chunk_from_event, is_terminal_event, round_from_event, state_from_event,
    state_to_execution_status, state_to_member_status, CHUNK_LLM_OUTPUT, CHUNK_LLM_REASONING,
    CHUNK_TOOL_CALL, CHUNK_TOOL_RESULT,
};
pub use native::{EnvModelResolver, ModelResolver, NativeMemberFactory, NATIVE_FACTORY_NAME};
