//! Harness rail trait and registry, plus the security, memory, skills, subagent, and evolution rails.
//!
//! Ports `openjiuwen/harness/rails/` and the hook model of
//! `openjiuwen/core/single_agent/rail/base.py`. [`HarnessRail`] is a single trait
//! covering the nine kernel lifecycle events of
//! [`openjiuwen_agent::AgentRail`] plus the four outer task-loop events of the
//! `DeepAgent` loop.
//!
//! Rails never call the model and never execute tools: they mutate the
//! [`HarnessRailContext`] handed to them, publish prompt sections into the
//! context's `extra` channel, and return control signals. The agent owns the
//! firing points; this module owns the contract.
//!
//! [`HarnessRailRegistry`] keeps the rails stable-sorted by descending
//! [`HarnessRail::priority`] behind a copy-on-write snapshot, so firing never
//! holds a lock and a registration issued during dispatch takes effect on the
//! next firing point. [`HarnessRailAdapter`] bridges a harness rail into the
//! kernel [`openjiuwen_agent::AgentRail`] slot of the inner `ReActAgent`.
//!
//! Built-in rails live in dedicated modules: `security`, `memory`, `skills`,
//! `subagent`, and `evolution` (the [`EvolutionRail`] seam, its adapter, and the
//! harness-side [`MetricsEvolutionRail`] with its metrics, proposal log and run
//! snapshot).
mod context;
mod evolution;
mod heartbeat;
mod kv_cache;
mod memory;
pub mod personal_context;
mod registry;
mod security;
mod sha256;
mod skills;
mod subagent;

pub use context::{
    last_user_text, prompt_sections, publish_prompt_section, HarnessRailContext, TaskIterationInfo,
    TaskIterationPhase, MEMORY_CONTEXT_EXTRA_KEY, MEMORY_QUERY_EXTRA_KEY,
    PROMPT_SECTIONS_EXTRA_KEY,
};
pub use evolution::{
    default_metrics_clock, evolution_adapter, EvolutionMetrics, EvolutionProposal, EvolutionRail,
    EvolutionRailAdapter, EvolutionSnapshot, IterationRecorder, MetricsClock, MetricsEvolutionRail,
    MetricsEvolutionRailConfig, NoopEvolutionRail, ProposalLog, EVOLUTION_SNAPSHOT_VERSION,
};
pub use heartbeat::{
    build_heartbeat_section, default_heartbeat_clock, heartbeat_prompt, HeartbeatClock,
    HeartbeatEvent, HeartbeatRail, HeartbeatRailConfig, HeartbeatReason,
    DEFAULT_HEARTBEAT_EVENT_CAPACITY, DEFAULT_HEARTBEAT_INTERVAL_MS, HEARTBEAT_RUN_KIND,
    HEARTBEAT_SECTION, HEARTBEAT_SECTION_PRIORITY, RUN_KIND_EXTRA_KEY,
};
pub use kv_cache::{
    affinity_enabled, is_sticky_subagent_type, resolve_sub_session_id,
    resolve_subagent_parent_cache_id, scope_sub_session_id, KvCacheAction, KvCacheAffinityConfig,
    KvCacheLifecycleRail, KvCacheLineage, KvCacheProviderRegistry, KvCacheSubagent,
    KvCacheSubagentLifecycle, KvCacheSubagentSession, KvCacheSubagentState, RuntimeSessionLineage,
    UnsupportedKvCacheBackend, AFFINITY_PARENT_SESSION_ID_ENV, STICKY_SUBAGENT_TYPE,
};
pub use memory::{
    LongTermMemoryAdapter, MemoryRail, MemoryRailConfig, MemoryRecall, MemoryWriteBack,
    RecalledMemory, TurnRecord, DEFAULT_MEMORY_SOURCE,
};
pub use personal_context::{
    messages_are_contiguous, normalize_source_locator, resolve_context_relative_path, search,
    source_id_for_locator, ContextEntry, FetchRequest, FetchedSource, IngestReport,
    IngestedDocument, LocalDirectoryConfig, LocalFilesProvider, PersonalContextRail,
    PersonalContextRailConfig, PersonalContextStatus, PersonalContextStore, SearchDocument,
    SearchHit, SourceKind, SourceLink, SourceLinkBook, SourceMetadata, SourceMetadataStore,
    SourceProvider, SourceProviderRegistry, UnsupportedSourceProvider,
};
pub use registry::{HarnessRailAdapter, HarnessRailRegistry};
pub use security::SecurityRail;
pub use skills::{SkillsRail, SkillsRailConfig};
pub use subagent::{
    SpawnRequest, SubAgentHandle, SubAgentSpawner, SubAgentState, SubAgentStatus, SubagentRail,
};

/// Dispatch priorities of the built-in rails.
///
/// The ladder mirrors the Python rail priorities, so registration order between
/// rails of different modules matches the reference implementation.
pub mod rail_priority {
    /// System-operation rail (`SysOperationRail`): first, it owns sandbox setup.
    pub const SYS_OPERATION: i32 = 100;
    /// MCP tool rail.
    pub const MCP: i32 = 95;
    /// Skill discovery/selection rail.
    pub const SKILLS: i32 = 95;
    /// Subagent prompt rail.
    pub const SUBAGENT: i32 = 95;
    /// Task planning rail.
    pub const PLANNING: i32 = 90;
    /// Interrupt (human-in-the-loop) rail.
    pub const INTERRUPT: i32 = 90;
    /// Guardrail/content-safety rail.
    pub const SECURITY: i32 = 90;
    /// Agent-mode switching rail.
    pub const AGENT_MODE: i32 = 85;
    /// Context-window/compression rail.
    pub const CONTEXT: i32 = 85;
    /// Safety prompt rail (`SafetyPromptRail`) — the prompt-side guardrails.
    pub const SAFETY_PROMPT: i32 = 85;
    /// Long-term memory rail.
    pub const MEMORY: i32 = 80;
    /// Heartbeat rail.
    pub const HEARTBEAT: i32 = 80;
    /// Model/tool resilience (retry/backup) rail.
    pub const RESILIENCE: i32 = 70;
    /// Self-evolution rail.
    pub const EVOLUTION: i32 = 60;
    /// LSP rail.
    pub const LSP: i32 = 60;
    /// Priority of a rail that overrides nothing (`DEFAULT_RAIL_PRIORITY`).
    pub const DEFAULT: i32 = 50;
    /// Completion-signal rail.
    pub const COMPLETION: i32 = 10;
    /// Mixture-of-agents rail.
    pub const MOA: i32 = 10;
}

/// Priority a rail gets when it does not override [`HarnessRail::priority`].
pub const DEFAULT_RAIL_PRIORITY: i32 = rail_priority::DEFAULT;

/// A hook the harness fires around its kernel and outer task-loop steps.
///
/// Every hook has a no-op default, so a rail overrides only what it needs. A
/// returned `Err` aborts the run for that firing point: the registry stops
/// dispatching and the error propagates to the caller.
///
/// The first nine hooks mirror [`openjiuwen_agent::AgentRail`]; the last four
/// are outer-loop events owned by the `DeepAgent` task loop.
#[async_trait::async_trait]
pub trait HarnessRail: Send + Sync {
    /// Stable name of the rail; registering two rails with one name replaces
    /// the first.
    fn name(&self) -> &str;

    /// Dispatch priority; higher runs first. Defaults to
    /// [`DEFAULT_RAIL_PRIORITY`].
    fn priority(&self) -> i32 {
        rail_priority::DEFAULT
    }

    /// A new invocation starts, before any message is built.
    async fn before_invoke(
        &self,
        _ctx: &mut HarnessRailContext,
    ) -> openjiuwen_common::error::Result<()> {
        Ok(())
    }

    /// The invocation finished and its output is final.
    async fn after_invoke(
        &self,
        _ctx: &mut HarnessRailContext,
    ) -> openjiuwen_common::error::Result<()> {
        Ok(())
    }

    /// About to call the model with the prepared window.
    async fn before_model_call(
        &self,
        _ctx: &mut HarnessRailContext,
    ) -> openjiuwen_common::error::Result<()> {
        Ok(())
    }

    /// The model returned a reply.
    async fn after_model_call(
        &self,
        _ctx: &mut HarnessRailContext,
    ) -> openjiuwen_common::error::Result<()> {
        Ok(())
    }

    /// The model call failed.
    async fn on_model_exception(
        &self,
        _ctx: &mut HarnessRailContext,
    ) -> openjiuwen_common::error::Result<()> {
        Ok(())
    }

    /// About to execute one tool call.
    async fn before_tool_call(
        &self,
        _ctx: &mut HarnessRailContext,
    ) -> openjiuwen_common::error::Result<()> {
        Ok(())
    }

    /// One tool call returned.
    async fn after_tool_call(
        &self,
        _ctx: &mut HarnessRailContext,
    ) -> openjiuwen_common::error::Result<()> {
        Ok(())
    }

    /// The tool call failed.
    async fn on_tool_exception(
        &self,
        _ctx: &mut HarnessRailContext,
    ) -> openjiuwen_common::error::Result<()> {
        Ok(())
    }

    /// One `ReAct` iteration finished (tool messages appended).
    async fn after_react_iteration(
        &self,
        _ctx: &mut HarnessRailContext,
    ) -> openjiuwen_common::error::Result<()> {
        Ok(())
    }

    /// A user message entered the outer loop (a new round or a follow-up).
    async fn on_user_message(
        &self,
        _ctx: &mut HarnessRailContext,
    ) -> openjiuwen_common::error::Result<()> {
        Ok(())
    }

    /// About to drain queued steering events into the running task.
    async fn before_steering_drain(
        &self,
        _ctx: &mut HarnessRailContext,
    ) -> openjiuwen_common::error::Result<()> {
        Ok(())
    }

    /// About to run one outer task-loop iteration ([`TaskIterationPhase::Before`]).
    async fn before_task_iteration(
        &self,
        _ctx: &mut HarnessRailContext,
    ) -> openjiuwen_common::error::Result<()> {
        Ok(())
    }

    /// One outer task-loop iteration finished ([`TaskIterationPhase::After`]).
    async fn after_task_iteration(
        &self,
        _ctx: &mut HarnessRailContext,
    ) -> openjiuwen_common::error::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn priority_ladder_matches_python() {
        assert_eq!(rail_priority::SYS_OPERATION, 100);
        assert_eq!(rail_priority::MCP, 95);
        assert_eq!(rail_priority::SKILLS, 95);
        assert_eq!(rail_priority::SUBAGENT, 95);
        assert_eq!(rail_priority::PLANNING, 90);
        assert_eq!(rail_priority::INTERRUPT, 90);
        assert_eq!(rail_priority::SECURITY, 90);
        assert_eq!(rail_priority::AGENT_MODE, 85);
        assert_eq!(rail_priority::CONTEXT, 85);
        assert_eq!(rail_priority::SAFETY_PROMPT, 85);
        assert_eq!(rail_priority::MEMORY, 80);
        assert_eq!(rail_priority::HEARTBEAT, 80);
        assert_eq!(rail_priority::RESILIENCE, 70);
        assert_eq!(rail_priority::EVOLUTION, 60);
        assert_eq!(rail_priority::LSP, 60);
        assert_eq!(rail_priority::DEFAULT, 50);
        assert_eq!(rail_priority::COMPLETION, 10);
        assert_eq!(rail_priority::MOA, 10);
        assert_eq!(DEFAULT_RAIL_PRIORITY, 50);
    }

    #[test]
    fn default_rail_priority_is_used_when_not_overridden() {
        struct PlainRail;

        #[async_trait::async_trait]
        impl HarnessRail for PlainRail {
            fn name(&self) -> &'static str {
                "plain"
            }
        }

        let rail = PlainRail;
        assert_eq!(rail.name(), "plain");
        assert_eq!(rail.priority(), rail_priority::DEFAULT);
    }
}
