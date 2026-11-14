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
//! | [`org`] | organisation roles, reporting lines, delegation rules, swarmflow, fork/compact |
//! | [`models`] | model pool, router and per-member allocation |
//! | [`kv_cache`] | KV-cache affinity identity and cleanup hooks |
//! | [`observability`] | `OTel` hook seam (no-op without the `otel` feature) |
//! | [`memory`] | team-memory wiring: lifecycle hooks, member tools, `TEAM_MEMORY.md` export/import |
//! | [`monitor`] | plain-text status stream for a live team |
//! | [`attach`] | cross-process attach over Unix domain sockets |
//!
//! Feature matrix (default features `["mcp"]`):
//!
//! | Feature | Default | Effect |
//! | --- | --- | --- |
//! | `mcp` | yes | the `openjiuwen-team-mcp` stdio server and its tool catalogue |
//! | `sqlite` | no | `SqliteTeamStore`, the persistent `TeamStore` backend |
//! | `otel` | no | structured `OTel` hooks and the in-memory span sink in [`observability`] |
//! | `otlp` | no | `OtlpMemoryObserver`: forwards team-memory events to `openjiuwen-extensions`' `OTel` sink |
//! | `distributed` | no | transport seams (ZeroMQ/Pulsar) that reject at construction |
//! | `live-tests` | no | opt-in live-network test gate |
//!
//! Capabilities: in-process, OS-process, external-CLI and SSH members; local and git-worktree
//! workspaces; the organisation layer (chart, delegation, org task pool, org mailbox, swarmflow,
//! fork/compact); model-pool routing and KV-cache affinity; observability hooks plus the monitor
//! status stream; cross-process attach over Unix domain sockets; and the `sqlite` team store.
//!
//! Genuinely deferred, each reached through a seam that returns
//! [`openjiuwen_common::error::Error::Unsupported`]: ZeroMQ/Pulsar transports, PostgreSQL/MySQL
//! stores, remote distributed workspace transport, OTLP span export beyond the team-memory bridge
//! (behind `otlp`), the TUI, authentication/TLS on attach, and vendor CLI harness translation.
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

pub mod attach;
pub mod constants;
pub mod experts;
pub mod harness;
pub mod interaction;
pub mod kv_cache;
pub mod mcp;
pub mod member;
pub mod memory;
pub mod messager;
pub mod models;
pub mod monitor;
pub mod observability;
pub mod org;
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
pub use storage::*;
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
pub use member::*;
// --- human-in-the-loop -----------------------------------------------------
pub use interaction::{
    approval_result_event, parse_confirm, parse_interact_str, resolve_targets, ConfirmPayload,
    DeliverResult, DeliverResult as InteractDeliverResult, GodViewMessage, GroupChatMessage,
    HumanAgentInbox, HumanAgentMessage, HumanAgentToolCall, InteractGate, InteractPayload,
    OperatorMessage, Route, ToolApprovalRequest,
};
// --- runtime ---------------------------------------------------------------
pub use runtime::{
    checkpoint_config, clear_global_registry, decide_run_action, global_registry,
    parse_initial_input, set_global_registry, try_global_registry, AttachConfig, AttachHost,
    BackendAttachHandler, CheckpointConfig, CheckpointStoreConfig, ForkDirective, InitialInput,
    LiveTeamState, MemberHealth, ModelAllocator, ModelSelection, OrgBinding, OrganizationBlock,
    RunAction, RunActionKind, SpecBuild, TeamAgent, TeamBackend, TeamBuildDeps, TeamHealth,
    TeamMemberSpawner, TeamModelAllocation, TeamModelRouter, TeamRuntimeManager, TeamRuntimePool,
    TeamSignals,
};
// --- harness protocol ------------------------------------------------------
pub use harness::{
    chunk_from_event, is_terminal_event, round_from_event, state_from_event,
    state_to_execution_status, state_to_member_status, CheckpointResume, EnvModelResolver,
    FileMemberCheckpointStore, HarnessMemberRuntime, InMemoryMemberCheckpointStore,
    MemberCheckpointPolicy, MemberCheckpointRecord, MemberCheckpointSink, MemberCheckpointStore,
    MemberHarnessContext, MemberHarnessFactory, ModelResolver, NativeCheckpoint,
    NativeMemberFactory,
};
// --- CLI and MCP -----------------------------------------------------------
pub use mcp::{build_mcp_server, TeamMcpServer, OPERATOR_INSTRUCTIONS, SERVER_NAME};
pub use skill::{
    connect_live_client, connect_socket_client, member_tools_for, operator_tools, run_skill_cli,
    ClaimStatus, CliCommand, ClientScope, CreateOutcome, DescriptorArgs, MemberCommand,
    MemberSummary, OpOutcome, OperatorCommand, SendOutcome, SkillCli, SocketTeamClient,
    StoreTeamClient, TaskCommand, TeamClient, TeamJoinDescriptor, TeamToolDef, ViewAction,
};
// --- organisation ----------------------------------------------------------
pub use org::*;
// --- model routing ---------------------------------------------------------
pub use models::*;
// --- team memory -----------------------------------------------------------
pub use memory::{
    memory_config, team_memory_tools, MemoryConfig, MemoryExtractorKind, MemoryStoreConfig,
    OtelSpanSink, TeamMemoryTool, TeamMemoryWiring, MEMORY_METADATA_KEY, TEAM_MEMORY_FILE,
};
// --- KV-cache affinity -----------------------------------------------------
pub use kv_cache::*;
// --- observability ---------------------------------------------------------
pub use observability::*;
// --- live status monitor ---------------------------------------------------
// Listed explicitly instead of `monitor::*`: the glob would re-export the `monitor::stream`
// module, which collides with the `schema::stream` module already re-exported above.
pub use monitor::{
    snapshot_events, BackpressurePolicy, EmitOutcome, MonitorEmitter, MonitorEvent,
    MonitorReceiver, StampedEvent, StatusStream, StatusStreamHandle, StreamSummary, TeamMonitor,
    TeamMonitorConfig, TimestampPolicy,
};
// --- cross-process attach --------------------------------------------------
// Listed explicitly instead of `attach::*`: the glob would re-export the `attach::protocol`
// module, which collides with the `member::protocol` module already re-exported above.
pub use attach::{
    AttachClient, AttachError, AttachHandler, AttachOp, AttachResult, AttachServer, RequestFrame,
    ResponseFrame, MAX_FRAME_BYTES, PROTOCOL_VERSION,
};
// The protocol crate owns this type; agent-teams re-exports it so hosts can branch on member
// state without depending on the protocol crate directly.
pub use openjiuwen_harness_protocol::HarnessState;

/// Returns the crate name.
#[must_use]
pub fn crate_name() -> &'static str {
    "openjiuwen-agent-teams"
}
