//! Handoff team: sequential agent-to-agent control transfer over the team bus.
//!
//! Python parity: `openjiuwen/core/multi_agent/teams/handoff/` in the openJiuwen
//! reference implementation. A handoff team runs one member at a time: that
//! member either answers, in which case the chain ends, or asks for a transfer,
//! in which case the next member continues with the accumulated history.
//!
//! ## Flow
//!
//! The strategy registers one endpoint per member — an agent whose id is
//! `__handoff_ep_{team_id}_{agent_id}` — and subscribes it to the pub/sub topic
//! `container_{agent_id}`. To start a chain it publishes a [`HandoffRequest`] to
//! the current member's endpoint. The endpoint's `ContainerAgent` then:
//!
//! 1. resolves its target through the runtime and mounts the targets'
//!    `transfer_to_<id>` tools on it;
//! 2. invokes the target in a child team session;
//! 3. writes the target's stream frames to the team stream and appends
//!    `{"agent", "output"}` to the request history;
//! 4. either completes the chain through the [`HandoffOrchestrator`], forwards a
//!    team interrupt, or publishes the next [`HandoffRequest`] hop.
//!
//! The coordinator enforces the hop cap, the termination condition, and the
//! route graph; every hop and the final result are keyed by the team session id
//! in a [`HandoffRegistry`], so a resumed chain picks up its coordinator state
//! from the session.
//!
//! This module currently holds the wave-1 pieces: configuration, the request
//! envelope, signal extraction, the transfer tool, interrupt handling, and the
//! coordinator. The container endpoint and the `HandoffStrategy` build on them.
// The wave-2 container/strategy modules and the crate-root re-export list (owned
// by another task) are the consumers of this module's public surface; until both
// land, rustc sees the items as unreachable, unused, and therefore dead. Remove
// this allow once `lib.rs` re-exports the handoff items.
#![allow(dead_code, unreachable_pub, unused_imports)]

/// Signal key carrying the id of the agent to hand off to.
pub const HANDOFF_TARGET_KEY: &str = "__handoff_to__";

/// Signal key carrying the context message passed to the next agent.
pub const HANDOFF_MESSAGE_KEY: &str = "__handoff_message__";

/// Signal key carrying the reason for the handoff.
pub const HANDOFF_REASON_KEY: &str = "__handoff_reason__";

/// Team-session state key holding the coordinator snapshot
/// (`current_agent_id`, `handoff_count`) for interrupt/resume.
pub const COORDINATOR_STATE_KEY: &str = "__handoff_coordinator__";

/// Team-session state key holding the accumulated hop history.
pub const HISTORY_STATE_KEY: &str = "__handoff_history__";

/// Index into the hop history marking where the current turn starts.
pub const CONTEXT_HISTORY_STATE_KEY: &str = "__handoff_ctx_history__";

mod config;
mod interrupt;
mod orchestrator;
mod request;
mod signal;
mod tool;

pub use config::{
    HandoffConfig, HandoffRoute, HandoffSnapshot, HandoffTeamConfig, TerminationCondition,
};
pub use interrupt::{extract_interrupt_signal, flush_team_session, TeamInterruptSignal};
pub use orchestrator::{HandoffOrchestrator, HandoffRegistry};
pub use request::{HandoffHistoryEntry, HandoffRequest};
pub use signal::{
    extract_handoff_signal, extract_handoff_signal_from_observations,
    extract_handoff_signal_from_session, HandoffSignal,
};
pub use tool::HandoffTool;
