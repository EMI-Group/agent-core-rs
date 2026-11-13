//! `DeepAgent` harness of the openJiuwen SDK — a port of the Python `openjiuwen.harness` package.
//!
//! This crate composes the `openjiuwen_core` kernel (agent runtime, sessions, sysop, security,
//! memory, context engine) into the `DeepAgent` task-loop framework with its workspace, permission
//! engine, coding tools, rails, prompts, skills and subagents.
//! It also implements the native provider for the provider-neutral SPI defined by
//! `openjiuwen_harness_protocol`.
//! The module tree and per-module responsibilities are documented in `CONTEXT.md`.
//!
//! # Flat API surface
//!
//! Every primary public item is also re-exported flat from the crate root, mirroring the Python
//! `openjiuwen.harness` package whose submodules are additionally flattened into `__init__`.
//! Module paths remain canonical: `openjiuwen_harness::agent::DeepAgent` and
//! `openjiuwen_harness::DeepAgent` name the same type, so consumers should prefer the module path
//! and treat the flat exports as a convenience spelling.
//!
//! Names that would collide are resolved with documented aliases:
//! - `tools::tool_names` keeps its name; the prompt-side list is exported as
//!   [`tool_description_names`] (alias of `prompts::tool_names`).
//! - `schema::SubAgentConfig` is the single flat spelling; `subagents` re-exports the same type,
//!   so it is listed once.
//! - `rails::SubAgentStatus` (rail bookkeeping) and `subagents::SubagentStatus` (runtime state)
//!   are distinct types and keep their distinct spellings.

#![forbid(unsafe_code)]

pub mod agent;
pub mod cli;
pub mod lsp;
pub mod manifest;
pub mod moa;
pub mod observability;
pub mod permission;
pub mod persistent_agent;
pub mod prompts;
pub mod protocol;
pub mod rails;
pub mod schema;
pub mod skills;
pub mod spec;
pub mod subagents;
pub mod task_loop;
pub mod tools;
pub mod watcher;
pub mod workspace;
pub mod worktree;

// --- agent -----------------------------------------------------------------
pub use agent::{
    apply_deep_agent_parts, clear_state, create_deep_agent, load_state, resolve_deep_agent_parts,
    save_state, DeepAgent, DeepAgentBuilder, DeepAgentParts, DeepAgentSubAgentFactory,
    SESSION_STATE_KEY,
};

// --- schema ----------------------------------------------------------------
pub use schema::{
    AgentMode, DeepAgentConfig, DeepAgentState, DeepLoopEvent, DeepLoopEventType,
    InputDispatchMode, InteractionEvent, InteractionPhase, OnMaxIterations, PlanModeState,
    PromptMode, RoundOutcome, RoundWorkItem, SendInputRequest, StopConditionEvaluator,
    StopConditionState, StopEvaluationContext, StopVerdict, SubAgentConfig,
    TaskLoopNoProgressGuardConfig, TaskPlan, TodoItem, TodoStatus,
};

// --- workspace -------------------------------------------------------------
pub use workspace::{
    is_safe_relative_path, DirectoryBuilder, DirectoryNode, Workspace, WorkspaceNode,
};

// --- permission ------------------------------------------------------------
pub use permission::{
    PermissionConfirmResponse, PermissionEngine, PermissionEngineConfig, PermissionHost,
    PermissionInterruptRail, PermissionLevel, PermissionResult, PermissionsSection,
};

// --- tools -----------------------------------------------------------------
/// Names of the thirty-two tools of [`BuiltinTool::EXTENDED`] (the default twenty plus the
/// twelve extended ones), in registry order.
pub use tools::extended_tool_names;
/// The goal rail, its tunables, and the `ctx.extra` key under which it queues its events.
///
/// [`GoalRail`] is a [`HarnessRail`] registered like the rails in [`rails`], but it is reachable
/// only as `tools::goal::GoalRail`: [`rails`] must not depend on [`tools`], so the goal rail is
/// exported from the module that owns it instead.
pub use tools::goal::{GoalRail, GoalRailConfig, GOAL_EVENT_EXTRA_KEY};
pub use tools::{
    tool_names, AudioMetadata, AudioMetadataArgs, AudioMetadataTool, BuiltinTool, CronAction,
    CronArgs, CronJob, CronSchedule, CronScheduler, CronTool, FakeMcpResourceProvider,
    FakeSearchBackend, FetchRequest, FetchResponse, FetchWebpageArgs, FetchWebpageTool,
    GetCurrentGoalArgs, GetCurrentGoalTool, GoalAssessment, GoalAssessmentStatus, GoalRecord,
    GoalStatus, GoalStore, HttpFetcher, ImageMetadata, ImageMetadataArgs, ImageMetadataTool,
    ListMcpResourcesTool, McpResource, McpResourceContent, McpResourceProvider, MediaKind,
    MemoryToolBackend, ReadMcpResourceArgs, ReadMcpResourceTool, ReqwestFetcher, SearchBackend,
    SearchResult, SubmitGoalReportArgs, SubmitGoalReportTool, TokenUsage, ToolIndex, ToolRegistry,
    ToolSearchArgs, ToolSearchHit, ToolSearchTool, UserAsker, WebSearchArgs, WebSearchTool,
};

// --- rails -----------------------------------------------------------------
pub use rails::{
    default_metrics_clock, evolution_adapter, rail_priority, EvolutionMetrics, EvolutionProposal,
    EvolutionRail, EvolutionRailAdapter, EvolutionSnapshot, HarnessRail, HarnessRailAdapter,
    HarnessRailContext, HarnessRailRegistry, IterationRecorder, MemoryRail, MetricsClock,
    MetricsEvolutionRail, MetricsEvolutionRailConfig, NoopEvolutionRail, ProposalLog, SecurityRail,
    SkillsRail, SubagentRail, TaskIterationInfo, EVOLUTION_SNAPSHOT_VERSION,
};

// --- lsp -------------------------------------------------------------------
pub use lsp::{
    ChannelTransport, LspClient, LspServerConfig, LspTool, LspToolConfig, LspTransport,
    StdioTransport,
};

// --- prompts ---------------------------------------------------------------
/// Alias of `prompts::tool_names` (the prompt-side tool list); the flat [`tool_names`] spelling
/// resolves to the `tools` registry order.
pub use prompts::tool_names as tool_description_names;
pub use prompts::{
    sanitize_path, sanitize_user_content, HarnessPromptBuilder, PromptReport, PromptSection,
    SectionName,
};

// --- subagents -------------------------------------------------------------
/// Retention bound of the persisted sub-agent bucket for `max_subagents` live instances.
pub use subagents::max_persisted_records;
/// Formats epoch milliseconds as an RFC 3339 UTC timestamp (the runtime's stamp format).
pub use subagents::rfc3339_from_ms;
/// Reason stamped on an instance closed by the LRU eviction policy.
pub use subagents::EVICTION_REASON;
/// Key of the persisted sub-agent bucket inside the session state.
pub use subagents::SUBAGENTS_KEY;
pub use subagents::{
    load_bucket_file, save_bucket_file, select_lru_candidate, subagent_tools,
    subagent_tools_with_transcript, Clock, LoadReport, LruCandidate, ManualClock,
    PersistedSubAgentBuckets, PersistedSubAgentRecord, ResumeReport, SpawnBudget, SubAgentFactory,
    SubAgentId, SubAgentOutcome, SubAgentPreset, SubAgentQuota, SubAgentRegistry, SubAgentRuntime,
    SubAgentSpawnRequest, SubAgentSpec, SubagentActivity, SubagentMetadata, SubagentStatus,
    SubagentTranscriptTool, SystemClock, TaskTool, TranscriptBuffer, TranscriptChannel,
    TranscriptEntry, TranscriptKind, TranscriptSubscription, TranscriptSummary,
    DEFAULT_TRANSCRIPT_CAPACITY, DEFAULT_TRANSCRIPT_CHANNEL_CAPACITY,
};

// --- skills ----------------------------------------------------------------
pub use skills::{Skill, SkillBudget, SkillLibrary, SkillResourceValidator, SkillRootMount};

// --- spec ------------------------------------------------------------------
pub use spec::{
    resolve, DeepAgentSpec, PluginManifest, PluginRegistry, ResolvedSpec, SpecBuildOptions,
    SpecEntry, SpecRegistry, SpecSource,
};

// --- task loop -------------------------------------------------------------
pub use task_loop::{
    DefaultTaskLoopEventHandler, EventManager, LoopCoordinator, LoopQueues, RoundExecutor,
    StateSink, TaskLoopConfig, TaskLoopController, TaskLoopEventHandler,
};

// --- protocol --------------------------------------------------------------
pub use protocol::{
    native_card, NativeCheckpointState, NativeHarness, NativeHarnessOptions, NativeHarnessProvider,
};

// --- cli -------------------------------------------------------------------
/// Re-export of the `clap` crate that [`cli::Cli`] derives from.
///
/// The parser surface is invisible through the derive alone: a host or an
/// integration test cannot call `Cli::parse_from`/`Cli::try_parse_from`
/// (`clap::Parser`) or build the grammar with `Cli::command()`
/// (`clap::CommandFactory`) without naming the `clap` crate itself, and a
/// different `clap` version would not see those trait impls at all.
/// Re-exporting the exact version this crate was built with lets downstream
/// crates (`openjiuwen-e2e`, the `openjiuwen-cli` binary, tests) assert the CLI
/// grammar, error kinds (`clap::error::ErrorKind`) and process exit codes
/// (`clap::Error::exit_code`) without adding a dependency of their own.
///
/// This is a build-and-test seam, not part of the SDK domain API: production
/// code should hand a parsed [`cli::Cli`] to [`cli::run_cli`] rather than
/// parsing arguments itself.
pub use clap;
pub use cli::{run, run_cli, Cli, CliOptions, CliPermissionHost, Command, OutputFormat, Repl};

// --- manifest --------------------------------------------------------------
pub use manifest::{
    builtin_elements, Catalog, ElementKind, HarnessElementDescriptor, InterfaceMethod,
};

// --- observability ---------------------------------------------------------
pub use observability::{NoopTelemetry, Observability, TelemetrySink};

// --- worktree --------------------------------------------------------------
pub use worktree::{
    slugify, GitRunner, ProcessGitRunner, WorktreeEvent, WorktreeInfo, WorktreeManager,
    WorktreeRecord, WorktreeTool,
};

/// Returns the crate name.
#[must_use]
pub fn crate_name() -> &'static str {
    "openjiuwen-harness"
}
