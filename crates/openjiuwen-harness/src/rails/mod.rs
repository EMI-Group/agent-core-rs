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
    DEFAULT_HEARTBEAT_EVENT_CAPACITY, DEFAULT_HEARTBEAT_INTERVAL_MS,
    DEFAULT_HEARTBEAT_SESSION_CAPACITY, HEARTBEAT_RUN_KIND, HEARTBEAT_SECTION,
    HEARTBEAT_SECTION_PRIORITY, RUN_KIND_EXTRA_KEY,
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

/// Cross-module composition of the heartbeat, KV-cache and personal-context rails.
///
/// Each rail is unit-tested beside its own module; these tests prove that they
/// compose in one [`HarnessRailRegistry`]: the prompt-section channel merges
/// their sections in render order, the temporary personal-context attachment is
/// cleared without disturbing a static section, the heartbeat queue is readable
/// from the registered handle, and the KV-cache lifecycle observes the subagent
/// tool payloads fired by the same registry.
#[cfg(test)]
mod integration_tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::needless_pass_by_value
    )]

    use std::path::Path;
    use std::sync::Arc;

    use async_trait::async_trait;
    use openjiuwen_agent::{RailEvent, ToolCallResult};
    use openjiuwen_common::error::Result;
    use openjiuwen_foundation::prompt::ToolCall;
    use openjiuwen_foundation::BaseMessage;
    use openjiuwen_session::AgentSession;
    use parking_lot::Mutex;
    use serde_json::{json, Value};

    use super::personal_context::{PERSONAL_CONTEXT_PRIORITY, PERSONAL_CONTEXT_SECTION};
    use super::*;

    /// The root description every personal-context fixture publishes.
    const DESCRIPTION: &str = "# Personal context\n\nMy notes live in `notes/`.\n";

    /// A [`KvCacheProviderRegistry`] that records every provider call.
    #[derive(Default)]
    struct RecordingRegistry {
        /// Recorded `"<operation>:<cache_id>"` lines, in call order.
        calls: Mutex<Vec<String>>,
    }

    impl RecordingRegistry {
        /// The recorded calls, in order.
        fn calls(&self) -> Vec<String> {
            self.calls.lock().clone()
        }

        /// Records one provider call.
        fn record(&self, operation: &str, cache_id: &str) {
            self.calls.lock().push(format!("{operation}:{cache_id}"));
        }
    }

    #[async_trait]
    impl KvCacheProviderRegistry for RecordingRegistry {
        fn backend(&self) -> &'static str {
            "recording"
        }

        async fn prepare(&self, cache_id: &str) -> Result<()> {
            self.record("prepare", cache_id);
            Ok(())
        }

        async fn suspend(&self, cache_id: &str) -> Result<()> {
            self.record("suspend", cache_id);
            Ok(())
        }

        async fn release(&self, cache_id: &str) -> Result<()> {
            self.record("release", cache_id);
            Ok(())
        }
    }

    /// The three composed rails plus the handles the assertions read.
    struct Suite {
        /// The shared registry every firing goes through.
        registry: HarnessRailRegistry,
        /// The registered heartbeat rail, for its event queue.
        heartbeat: Arc<HeartbeatRail>,
        /// The lifecycle the KV-cache rail drives.
        lifecycle: Arc<KvCacheSubagentLifecycle>,
        /// The provider seam the lifecycle calls.
        providers: Arc<RecordingRegistry>,
    }

    /// Writes the switch and description of `home` and returns the rail.
    fn personal_rail(home: &Path, enabled: bool, description: &str) -> PersonalContextRail {
        let rail = PersonalContextRail::new(PersonalContextRailConfig::new(home));
        let switch = if enabled {
            "agent_use_enabled: true\n"
        } else {
            "agent_use_enabled: false\n"
        };
        std::fs::write(rail.config().config_path(), switch).unwrap();
        std::fs::create_dir_all(rail.context_root()).unwrap();
        std::fs::write(rail.description_path(), description).unwrap();
        rail
    }

    /// Registers the heartbeat, KV-cache and personal-context rails in one registry.
    fn compose(home: &Path, switch_enabled: bool, affinity: bool) -> Suite {
        let heartbeat = Arc::new(HeartbeatRail::new(HeartbeatRailConfig::default()));
        let providers = Arc::new(RecordingRegistry::default());
        let provider: Arc<dyn KvCacheProviderRegistry> = providers.clone();
        let lifecycle = Arc::new(KvCacheSubagentLifecycle::new(provider).with_config(
            KvCacheAffinityConfig {
                enable_kv_cache_affinity: affinity,
            },
        ));
        let kv_cache: Arc<dyn HarnessRail> =
            Arc::new(KvCacheLifecycleRail::new(Arc::clone(&lifecycle)));
        let personal: Arc<dyn HarnessRail> =
            Arc::new(personal_rail(home, switch_enabled, DESCRIPTION));
        let heartbeat_rail: Arc<dyn HarnessRail> = heartbeat.clone();

        let registry = HarnessRailRegistry::new();
        registry.register(heartbeat_rail);
        registry.register(kv_cache);
        registry.register(personal);
        Suite {
            registry,
            heartbeat,
            lifecycle,
            providers,
        }
    }

    /// A `BeforeModelCall` context with a contiguous one-message window.
    fn model_ctx(agent_id: &str, session_id: &str, run_kind: Option<&str>) -> HarnessRailContext {
        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeModelCall);
        ctx.base.agent_id = agent_id.to_string();
        ctx.base.session = Some(Arc::new(AgentSession::new(session_id)));
        ctx.base.messages = vec![BaseMessage::user("status report")];
        if let Some(run_kind) = run_kind {
            ctx.extra_mut()
                .insert(RUN_KIND_EXTRA_KEY.to_string(), json!(run_kind));
        }
        ctx
    }

    /// An `AfterToolCall` context around one call/result pair.
    fn tool_ctx(
        session_id: &str,
        name: &str,
        arguments: Value,
        payload: Value,
    ) -> HarnessRailContext {
        let call = ToolCall {
            name: name.to_string(),
            arguments: arguments.to_string(),
            ..ToolCall::default()
        };
        let mut ctx = HarnessRailContext::for_event(RailEvent::AfterToolCall);
        ctx.base.agent_id = "agent-a".to_string();
        ctx.base.session = Some(Arc::new(AgentSession::new(session_id)));
        ctx.base.tool_call = Some(call.clone());
        ctx.base.tool_result = Some(ToolCallResult::success(call, payload, String::new()));
        ctx
    }

    /// A spawn payload for the sticky `verification_agent`.
    fn spawn_payload() -> Value {
        json!({
            "success": true,
            "data": {"subagent_id": "sub-1", "sub_session_id": "sub-1"}
        })
    }

    #[tokio::test]
    async fn one_model_call_publishes_the_heartbeat_and_personal_context_sections() {
        let home = tempfile::tempdir().unwrap();
        let suite = compose(home.path(), true, true);

        let mut ctx = model_ctx("agent-a", "session-a", Some(HEARTBEAT_RUN_KIND));
        suite.registry.fire(&mut ctx).await.unwrap();

        // Sections render in `(priority, name)` order: 40 before 78.
        let sections = prompt_sections(&ctx);
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].name, PERSONAL_CONTEXT_SECTION);
        assert_eq!(sections[0].priority, PERSONAL_CONTEXT_PRIORITY);
        assert_eq!(sections[1].name, HEARTBEAT_SECTION);
        assert_eq!(sections[1].priority, HEARTBEAT_SECTION_PRIORITY);

        let personal = sections[0].content.get("cn").unwrap();
        assert!(personal.contains("# 主动上下文"), "wrapper heading is kept");
        assert!(
            personal.contains("My notes live in `notes/`."),
            "the raw description is attached"
        );
        let heartbeat = sections[1].content.get("cn").unwrap();
        assert!(heartbeat.contains("HEARTBEAT_OK"));

        // The marked run queued exactly one event naming the agent and session.
        let events = suite.heartbeat.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].reason, HeartbeatReason::RunKind);
        assert_eq!(events[0].agent_id, "agent-a");
        assert_eq!(events[0].session_id, "session-a");
        assert_eq!(events[0].sequence, 1);
        // No provider call happened on the model-call firing.
        assert_eq!(suite.providers.calls().len(), 0);
    }

    #[tokio::test]
    async fn after_model_call_clears_only_the_temporary_section() {
        let home = tempfile::tempdir().unwrap();
        let suite = compose(home.path(), true, true);

        let mut ctx = model_ctx("agent-a", "session-a", Some(HEARTBEAT_RUN_KIND));
        suite.registry.fire(&mut ctx).await.unwrap();
        assert_eq!(prompt_sections(&ctx).len(), 2);

        ctx.base.event = RailEvent::AfterModelCall;
        suite.registry.fire(&mut ctx).await.unwrap();

        let sections = prompt_sections(&ctx);
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].name, HEARTBEAT_SECTION);
        assert_eq!(sections[0].priority, HEARTBEAT_SECTION_PRIORITY);
        // The heartbeat section is static: the firing count does not change.
        assert_eq!(suite.heartbeat.events().len(), 1);
    }

    #[tokio::test]
    async fn the_kv_cache_lifecycle_tracks_spawn_wait_and_close() {
        let home = tempfile::tempdir().unwrap();
        let suite = compose(home.path(), true, true);

        // A model-call firing leaves the provider seam untouched.
        let mut model = model_ctx("agent-a", "session-a", None);
        suite.registry.fire(&mut model).await.unwrap();
        assert_eq!(suite.providers.calls().len(), 0);

        // A successful sticky spawn prepares the child cache identity.
        let mut spawn = tool_ctx(
            "session-a",
            "subagent_spawn",
            json!({"subagent_type": "verification_agent"}),
            spawn_payload(),
        );
        suite.registry.fire(&mut spawn).await.unwrap();
        assert_eq!(suite.providers.calls(), vec!["prepare:session-a"]);
        let session = suite.lifecycle.get("sub-1").unwrap();
        assert_eq!(session.subagent_type, "verification_agent");
        assert_eq!(session.state, KvCacheSubagentState::Active);
        assert_eq!(session.actions, vec![KvCacheAction::Prepare]);

        // A terminal success suspends the sticky identity instead of releasing it.
        let mut wait = tool_ctx(
            "session-a",
            "subagent_wait",
            json!({"subagent_ids": ["sub-1"]}),
            json!({"success": true, "data": {"statuses": {"sub-1": "completed"}}}),
        );
        suite.registry.fire(&mut wait).await.unwrap();
        assert_eq!(
            suite.providers.calls(),
            vec!["prepare:session-a", "suspend:session-a"]
        );
        assert_eq!(
            suite.lifecycle.get("sub-1").unwrap().state,
            KvCacheSubagentState::Suspended
        );

        // Closing the subagent releases the identity.
        let mut close = tool_ctx(
            "session-a",
            "subagent_close",
            json!({"subagent_id": "sub-1"}),
            json!({"success": true, "data": {"subagent_id": "sub-1"}}),
        );
        suite.registry.fire(&mut close).await.unwrap();
        assert_eq!(
            suite.providers.calls(),
            vec![
                "prepare:session-a",
                "suspend:session-a",
                "release:session-a"
            ]
        );
        assert_eq!(
            suite.lifecycle.get("sub-1").unwrap().state,
            KvCacheSubagentState::Released
        );
    }

    #[tokio::test]
    async fn an_affinity_off_lifecycle_makes_no_provider_call() {
        let home = tempfile::tempdir().unwrap();
        let suite = compose(home.path(), true, false);

        let mut spawn = tool_ctx(
            "session-a",
            "subagent_spawn",
            json!({"subagent_type": "verification_agent"}),
            spawn_payload(),
        );
        suite.registry.fire(&mut spawn).await.unwrap();
        let mut wait = tool_ctx(
            "session-a",
            "subagent_wait",
            json!({"subagent_ids": ["sub-1"]}),
            json!({"success": true, "data": {"statuses": {"sub-1": "completed"}}}),
        );
        suite.registry.fire(&mut wait).await.unwrap();

        assert_eq!(suite.providers.calls().len(), 0);
        assert_eq!(suite.lifecycle.sessions().len(), 0);
    }

    #[tokio::test]
    async fn a_disabled_switch_publishes_only_the_heartbeat_section() {
        let home = tempfile::tempdir().unwrap();
        let suite = compose(home.path(), false, true);

        let mut ctx = model_ctx("agent-a", "session-a", Some(HEARTBEAT_RUN_KIND));
        suite.registry.fire(&mut ctx).await.unwrap();

        let sections = prompt_sections(&ctx);
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].name, HEARTBEAT_SECTION);
        assert_eq!(suite.heartbeat.events().len(), 1);
    }
}
