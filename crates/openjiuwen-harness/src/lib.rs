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
pub mod manifest;
pub mod observability;
pub mod permission;
pub mod prompts;
pub mod protocol;
pub mod rails;
pub mod schema;
pub mod skills;
pub mod subagents;
pub mod task_loop;
pub mod tools;
pub mod workspace;

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
pub use tools::{tool_names, BuiltinTool, MemoryToolBackend, ToolRegistry, UserAsker};

// --- rails -----------------------------------------------------------------
pub use rails::{
    rail_priority, EvolutionRail, HarnessRail, HarnessRailAdapter, HarnessRailContext,
    HarnessRailRegistry, MemoryRail, NoopEvolutionRail, SecurityRail, SkillsRail, SubagentRail,
    TaskIterationInfo,
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
pub use subagents::{
    subagent_tools, SubAgentFactory, SubAgentId, SubAgentOutcome, SubAgentPreset, SubAgentRegistry,
    SubAgentRuntime, SubAgentSpawnRequest, SubAgentSpec, SubagentStatus, TaskTool,
};

// --- skills ----------------------------------------------------------------
pub use skills::{Skill, SkillBudget, SkillLibrary, SkillResourceValidator, SkillRootMount};

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
pub use cli::{run, run_cli, Cli, CliOptions, CliPermissionHost, Command, OutputFormat, Repl};

// --- manifest --------------------------------------------------------------
pub use manifest::{
    builtin_elements, Catalog, ElementKind, HarnessElementDescriptor, InterfaceMethod,
};

// --- observability ---------------------------------------------------------
pub use observability::{NoopTelemetry, Observability, TelemetrySink};

/// Returns the crate name.
#[must_use]
pub fn crate_name() -> &'static str {
    "openjiuwen-harness"
}
