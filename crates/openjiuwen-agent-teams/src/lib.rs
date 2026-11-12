//! Leader/teammate multi-agent runtime.
//!
//! Maps to the Python package `openjiuwen.agent_teams` in the openJiuwen reference implementation:
//! team specs and the expert catalog, the in-process leader/teammate runtime (organisation
//! dispatch, task board, mailbox, member lifecycle, local workspace isolation), HITL forwarding,
//! the harness-protocol adapter, the `team-member` skill CLI and the `openjiuwen-team-mcp` server.
//!
//! The crate is layered bottom-up; each module owns its node of the Context Tree and documents
//! its own API surface:
//!
//! | Module | Responsibility |
//! | --- | --- |
//! | [`constants`] | reserved member names, defaults, mention parsing |
//! | [`schema`] | specs, team runtime types, statuses, task/event/stream records |
//! | [`experts`] | `ExpertCatalog` seam and agent-group packages |
//! | [`storage`] | `TeamStore` seam and the in-memory store |
//! | [`taskboard`] | task tree, dependency graph, claims, reviews |
//! | [`messager`] | `Messager` seam and the in-process transport |
//! | [`workspace`] | member workspace allocation (local directories) |
//! | [`member`] | member runtime seam, status machine, spawn handles |
//! | [`interaction`] | HITL payloads, routing grammar, interact gate |
//! | [`runtime`] | `TeamAgent` leader/teammate loop, dispatch, runtime manager |
//! | [`harness`] | harness-protocol adapter and native member factory |
//! | [`skill`] | the `team-member` skill CLI and join descriptor |
//! | [`mcp`] | the `openjiuwen-team-mcp` MCP server |
//!
//! Everything is in-process and tokio-native; distributed transports, OS-process members and
//! external CLI members are trait seams that return [`openjiuwen_common::error::Error::Unsupported`].
//!
//! ```
//! use openjiuwen_agent_teams::{is_reserved_name, parse_mention, DEFAULT_LEADER_MEMBER_NAME};
//!
//! assert_eq!(DEFAULT_LEADER_MEMBER_NAME, "team_leader");
//! assert!(is_reserved_name("user"));
//! assert_eq!(parse_mention("@dev-1 start"), Some(("dev-1", "start")));
//! ```
#![forbid(unsafe_code)]
// Python package names (`agent_teams`, `team_agent`) repeat deliberately in module names.
#![allow(clippy::module_name_repetitions)]

pub mod constants;
pub mod experts;
pub mod harness;
pub mod interaction;
pub mod mcp;
pub mod member;
pub mod messager;
pub mod runtime;
pub mod schema;
pub mod skill;
pub mod storage;
pub mod taskboard;
pub mod workspace;

// --- identity, defaults and addressing -------------------------------------
pub use constants::{
    is_reserved_name, parse_mention, BROADCAST_TARGET, DEFAULT_LANGUAGE,
    DEFAULT_LEADER_DISPLAY_NAME, DEFAULT_LEADER_MEMBER_NAME, DEFAULT_TEAM_NAME,
    HUMAN_AGENT_MEMBER_NAME, MAX_MESSAGE_CONTENT_CHARS, OPENJIUWEN_HOME_ENV, RESERVED_MEMBER_NAMES,
    TEAM_JOIN_ENV, USER_PSEUDO_MEMBER_NAME,
};
// --- data model ------------------------------------------------------------
pub use schema::*;
// --- expert catalog --------------------------------------------------------
pub use experts::{
    bind_group, clear_process_expert_catalog, compose_template_member, get_process_expert_catalog,
    load_agent_group, render_persona, set_process_expert_catalog, AgentGroupCard,
    AgentGroupMemberCard, AgentTemplateCard, BoundAgentGroup, BoundGroupMember, ExpertCatalog,
    ExpertRef, LoadedAgentGroupPackage, LocalizedText, PackageLocation,
};
// --- persistence -----------------------------------------------------------
pub use storage::{
    open_store, InMemoryTeamStore, MemberPatch, MemberRecord, MessageRecord, TeamInfoRecord,
    TeamStore,
};
// --- task board ------------------------------------------------------------
pub use taskboard::{
    detect_cycle, ReviewTally, TaskBoard, TaskBoardLimits, TaskMutation, TaskOpResult,
};
// --- messaging -------------------------------------------------------------
pub use messager::{
    create_messager, EventSink, InProcessMessager, Messager, MessagerPeerConfig,
    MessagerTransportConfig,
};
// --- workspaces ------------------------------------------------------------
pub use workspace::*;
// --- members ---------------------------------------------------------------
pub use member::{
    IdleDebounce, InProcessSpawnHandle, MemberActivityRegistry, MemberObserver, MemberRuntime,
    MemberSpawnRequest, MemberSpawner, RecordingObserver, SpawnRegistry, TeamMember,
};
// --- human-in-the-loop -----------------------------------------------------
pub use interaction::{
    approval_result_event, parse_confirm, parse_interact_str, resolve_targets, ConfirmPayload,
    DeliverResult, DeliverResult as InteractDeliverResult, GodViewMessage, GroupChatMessage,
    HumanAgentInbox, HumanAgentMessage, HumanAgentToolCall, InteractGate, InteractPayload,
    OperatorMessage, Route, ToolApprovalRequest,
};
// --- runtime ---------------------------------------------------------------
pub use runtime::{
    clear_global_registry, decide_run_action, global_registry, parse_initial_input,
    set_global_registry, try_global_registry, InitialInput, LiveTeamState, ModelAllocator,
    RunAction, RunActionKind, SpecBuild, TeamAgent, TeamBackend, TeamBuildDeps, TeamMemberSpawner,
    TeamModelAllocation, TeamRuntimeManager, TeamRuntimePool,
};
// --- harness protocol ------------------------------------------------------
pub use harness::{
    chunk_from_event, is_terminal_event, round_from_event, state_from_event,
    state_to_execution_status, state_to_member_status, EnvModelResolver, HarnessMemberRuntime,
    MemberHarnessContext, MemberHarnessFactory, ModelResolver, NativeCheckpoint,
    NativeMemberFactory,
};
// --- CLI and MCP -----------------------------------------------------------
pub use mcp::{build_mcp_server, TeamMcpServer, OPERATOR_INSTRUCTIONS, SERVER_NAME};
pub use skill::{
    connect_live_client, member_tools_for, operator_tools, run_skill_cli, ClaimStatus, CliCommand,
    ClientScope, CreateOutcome, DescriptorArgs, MemberCommand, MemberSummary, OpOutcome,
    OperatorCommand, SendOutcome, SkillCli, StoreTeamClient, TaskCommand, TeamClient,
    TeamJoinDescriptor, TeamToolDef, ViewAction,
};
// The protocol crate owns this type; agent-teams re-exports it so hosts can branch on member
// state without depending on the protocol crate directly.
pub use openjiuwen_harness_protocol::HarnessState;

/// Returns the crate name.
#[must_use]
pub fn crate_name() -> &'static str {
    "openjiuwen-agent-teams"
}
