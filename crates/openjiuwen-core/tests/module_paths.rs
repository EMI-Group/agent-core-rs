//! Proves the kernel umbrella's Python-shaped module paths resolve, and that each
//! alias points at the very same item as the concrete crate it re-exports.
//!
//! Every type check is a compile-time identity check (`TypeId`) rather than a string
//! comparison, so a rename or a diverging re-export inside a concrete crate fails
//! this test instead of silently drifting. Traits have no `TypeId`, so the trait
//! seams are proved by generic bounds and trait-object coercions instead.

use std::any::TypeId;

use openjiuwen_core::application::{
    create_llm_agent_config, ConstrainConfig, DefaultResponse, LLMAgent, LLMAgentConfig,
    ModelConfig, ModelInfo, WorkflowAgent, WorkflowAgentConfig,
};
use openjiuwen_core::common::error::Error;
use openjiuwen_core::context_engine::ContextEngine;
use openjiuwen_core::foundation::llm::Model;
use openjiuwen_core::foundation::prompt::BaseMessage;
use openjiuwen_core::graph::CompiledGraph;
use openjiuwen_core::memory::LongTermMemory;
use openjiuwen_core::multi_agent::BaseTeam;
use openjiuwen_core::retrieval::Document;
use openjiuwen_core::runner::{
    CallbackEvent, InMemoryMessageQueue, MessageQueue, ResourceMgr, RunOptions, Runner,
};
use openjiuwen_core::security::GuardrailPipeline;
use openjiuwen_core::session::{Session, SessionConfig};
use openjiuwen_core::single_agent::{AbilityManager, AgentCard, ReActAgent, ReActAgentConfig};
use openjiuwen_core::sys_operation::SysOperation;
use openjiuwen_core::workflow::Workflow;

/// Asserts at compile time that the umbrella alias and the concrete-crate item are
/// the same type.
fn assert_same_type<T: ?Sized + 'static, U: ?Sized + 'static>() {
    assert_eq!(TypeId::of::<T>(), TypeId::of::<U>());
}

/// Asserts at compile time that `T` satisfies the umbrella's controller executor
/// seam. A trait has no `TypeId`, so the bound is what proves identity: the
/// concrete crate's `dyn TaskExecutor` only satisfies the umbrella's bound when
/// both paths name the same trait.
fn assert_task_executor<T: openjiuwen_core::controller::TaskExecutor + ?Sized>() {}

/// Asserts at compile time that `T` satisfies the umbrella's query-rewriter seam.
fn assert_query_rewriter<T: openjiuwen_core::retrieval::rewrite::QueryRewriter + ?Sized>() {}
/// Asserts at compile time that `T` satisfies the umbrella's compression-state sink
/// seam.
fn assert_compression_state_sink<
    T: openjiuwen_core::context_engine::telemetry::CompressionStateSink + ?Sized,
>() {
}
/// Asserts at compile time that `T` satisfies the umbrella's team-memory store seam.
fn assert_team_memory_store<T: openjiuwen_core::memory::team::TeamMemoryStore + ?Sized>() {}
/// Asserts at compile time that `T` satisfies the umbrella's external memory-provider
/// seam. This is `openjiuwen_memory::external::MemoryProvider`, not the agent crate's
/// `memory_provider` module of the same name.
fn assert_memory_provider<T: openjiuwen_core::memory::external::MemoryProvider + ?Sized>() {}

/// References (but never calls) an item whose fn-item type is unnameable.
fn reference_item<T>(_item: T) {}

/// `install_trace_sink` as a fn pointer, so the alias and the concrete-crate path
/// can be compared by address.
type InstallTraceSinkFn = fn(
    &dyn openjiuwen_session::Session,
    std::sync::Arc<dyn openjiuwen_session::TraceSink>,
) -> Result<(), openjiuwen_common::error::Error>;

/// `diff_snapshots` as a fn pointer.
type DiffSnapshotsFn = fn(
    &openjiuwen_session::SessionSnapshot,
    &openjiuwen_session::SessionSnapshot,
) -> Vec<openjiuwen_session::JsonChange>;
/// `memory_provider::provider_tools` as a fn pointer.
type ProviderToolsFn = fn(
    std::sync::Arc<dyn openjiuwen_memory::MemoryProvider>,
) -> Vec<std::sync::Arc<dyn openjiuwen_foundation::Tool>>;
/// `loaders::fixtures::build_zip` as a fn pointer.
type BuildZipFn = fn(&[(&str, &str)]) -> openjiuwen_common::error::Result<Vec<u8>>;
/// `loaders::fixtures::pptx_slides` as a fn pointer.
type PptxSlidesFn = fn(&[(&str, &[&str])]) -> openjiuwen_common::error::Result<Vec<u8>>;

#[test]
fn common_paths_resolve() {
    assert_same_type::<Error, openjiuwen_common::error::Error>();
    assert_eq!(
        openjiuwen_core::common::CRATE_NAME,
        "openjiuwen-common",
        "the common alias forwards the concrete crate"
    );
}

#[test]
fn foundation_paths_resolve() {
    assert_same_type::<dyn Model, dyn openjiuwen_foundation::llm::Model>();
    assert_same_type::<BaseMessage, openjiuwen_foundation::prompt::BaseMessage>();
}

#[test]
fn session_paths_resolve() {
    assert_same_type::<dyn Session, dyn openjiuwen_session::Session>();
    assert_same_type::<SessionConfig, openjiuwen_session::SessionConfig>();
}

#[test]
fn graph_paths_resolve() {
    assert_same_type::<CompiledGraph, openjiuwen_graph::CompiledGraph>();
}

#[test]
fn workflow_paths_resolve() {
    assert_same_type::<Workflow, openjiuwen_workflow::Workflow>();
}

#[test]
fn context_engine_paths_resolve() {
    assert_same_type::<ContextEngine, openjiuwen_context::ContextEngine>();
}

/// Proves the `context_engine` alias forwards the context crate's `usage`,
/// `telemetry`, `policy`, `tokenizers`, and `processor::rule_compression` submodules
/// unchanged.
#[test]
fn context_engine_modules_paths_resolve() {
    use openjiuwen_core::context_engine::{
        policy::{
            lookup_model_default_window, ContextWindowPolicy, OpenRouterWindowCache,
            ResolvedContextWindow,
        },
        processor::rule_compression::{
            extract_query_terms, ContentRouter, RuleCompressionPipeline, RuleContentRouter,
        },
        telemetry::{CompressionStatePhase, RecordingSink},
        tokenizers::{
            artifacts::TokenizerArtifactManager, load_native_tokenizer, ModelEncoding,
            ModelTokenCounter, TokenizerRegistry, TokenizerSpec,
        },
        usage::{ContextUsageAnalyzer, ContextUsageSnapshot, SessionKVCacheAggregator},
    };

    // Usage telemetry: the analyzer, its snapshot shape, and the KV-cache rollup.
    assert_same_type::<ContextUsageAnalyzer, openjiuwen_context::usage::ContextUsageAnalyzer>();
    assert_same_type::<ContextUsageSnapshot, openjiuwen_context::usage::ContextUsageSnapshot>();
    assert_same_type::<SessionKVCacheAggregator, openjiuwen_context::usage::SessionKVCacheAggregator>(
    );

    // Compression-state telemetry. `CompressionStateSink` is a trait: prove the
    // concrete trait object satisfies the umbrella trait's bound.
    assert_same_type::<CompressionStatePhase, openjiuwen_context::telemetry::CompressionStatePhase>(
    );
    assert_same_type::<RecordingSink, openjiuwen_context::telemetry::RecordingSink>();
    assert_compression_state_sink::<dyn openjiuwen_context::telemetry::CompressionStateSink>();

    // Window resolution.
    assert_same_type::<ContextWindowPolicy, openjiuwen_context::policy::ContextWindowPolicy>();
    assert_same_type::<ResolvedContextWindow, openjiuwen_context::policy::ResolvedContextWindow>();
    assert_same_type::<OpenRouterWindowCache, openjiuwen_context::policy::OpenRouterWindowCache>();
    let core_lookup: fn(&str) -> Option<usize> = lookup_model_default_window;
    let concrete_lookup: fn(&str) -> Option<usize> =
        openjiuwen_context::policy::lookup_model_default_window;
    assert!(std::ptr::eq(
        core_lookup as *const (),
        concrete_lookup as *const ()
    ));

    // Tokenizer selection and offline artefacts.
    assert_same_type::<TokenizerRegistry, openjiuwen_context::tokenizers::TokenizerRegistry>();
    assert_same_type::<TokenizerSpec, openjiuwen_context::tokenizers::TokenizerSpec>();
    assert_same_type::<ModelTokenCounter, openjiuwen_context::tokenizers::ModelTokenCounter>();
    assert_same_type::<ModelEncoding, openjiuwen_context::tokenizers::ModelEncoding>();
    assert_same_type::<
        TokenizerArtifactManager,
        openjiuwen_context::tokenizers::artifacts::TokenizerArtifactManager,
    >();

    // The native-tokenizer loader takes `impl AsRef<Path>`, so its fn item is generic
    // and cannot be coerced to a higher-ranked `fn(&Path)` pointer. Pin one concrete
    // instantiation (`PathBuf`) through both paths and compare the addresses. Never
    // called.
    let core_load: fn(
        std::path::PathBuf,
    ) -> openjiuwen_common::error::Result<
        openjiuwen_context::tokenizers::SelectedTokenizer,
    > = load_native_tokenizer;
    let concrete_load: fn(
        std::path::PathBuf,
    ) -> openjiuwen_common::error::Result<
        openjiuwen_context::tokenizers::SelectedTokenizer,
    > = openjiuwen_context::tokenizers::load_native_tokenizer;
    assert!(std::ptr::eq(
        core_load as *const (),
        concrete_load as *const ()
    ));

    // Rule-based tool-result compression. `ContentRouter` is a type alias for
    // `RuleContentRouter`, so it is compared by `TypeId`, not by a trait bound.
    assert_same_type::<
        RuleContentRouter,
        openjiuwen_context::processor::rule_compression::RuleContentRouter,
    >();
    assert_same_type::<
        RuleCompressionPipeline,
        openjiuwen_context::processor::rule_compression::RuleCompressionPipeline,
    >();
    assert_same_type::<
        ContentRouter,
        openjiuwen_context::processor::rule_compression::RuleContentRouter,
    >();
    let core_terms: fn(&str, Option<&str>, Option<&str>) -> std::collections::BTreeSet<String> =
        extract_query_terms;
    let concrete_terms: fn(&str, Option<&str>, Option<&str>) -> std::collections::BTreeSet<String> =
        openjiuwen_context::processor::rule_compression::extract_query_terms;
    assert!(std::ptr::eq(
        core_terms as *const (),
        concrete_terms as *const ()
    ));
}

#[test]
fn memory_paths_resolve() {
    assert_same_type::<LongTermMemory, openjiuwen_memory::LongTermMemory>();
}

#[test]
fn retrieval_paths_resolve() {
    assert_same_type::<Document, openjiuwen_retrieval::Document>();
}

#[test]
fn sys_operation_paths_resolve() {
    assert_same_type::<SysOperation, openjiuwen_sysop::SysOperation>();
}

#[test]
fn security_paths_resolve() {
    assert_same_type::<GuardrailPipeline, openjiuwen_security::GuardrailPipeline>();
}

#[test]
fn single_agent_paths_resolve() {
    assert_same_type::<ReActAgent, openjiuwen_agent::ReActAgent>();
    assert_same_type::<ReActAgentConfig, openjiuwen_agent::ReActAgentConfig>();
    assert_same_type::<AgentCard, openjiuwen_agent::AgentCard>();
    assert_same_type::<AbilityManager, openjiuwen_agent::AbilityManager>();
}

#[test]
fn application_paths_resolve() {
    assert_same_type::<LLMAgent, openjiuwen_agent::LlmAgent>();
    assert_same_type::<LLMAgentConfig, openjiuwen_agent::LlmAgentConfig>();
    assert_same_type::<WorkflowAgent, openjiuwen_agent::WorkflowAgent>();
    assert_same_type::<WorkflowAgentConfig, openjiuwen_agent::WorkflowAgentConfig>();
    assert_same_type::<DefaultResponse, openjiuwen_agent::DefaultResponse>();
    assert_same_type::<ConstrainConfig, openjiuwen_agent::ConstrainConfig>();
    assert_same_type::<ModelConfig, openjiuwen_agent::ModelConfig>();
    assert_same_type::<ModelInfo, openjiuwen_agent::ModelInfo>();

    // The config helper is a function, so prove it is callable through the alias and
    // returns the re-exported config type.
    let config: LLMAgentConfig = create_llm_agent_config("app_agent", ModelConfig::default());
    assert_eq!(config.id, "app_agent");
}

#[test]
fn multi_agent_paths_resolve() {
    assert_same_type::<BaseTeam, openjiuwen_multiagent::BaseTeam>();
}

/// Asserts at compile time that `T` implements the runner's queue seam. A trait has
/// no `TypeId`, so the bound (together with the `&dyn MessageQueue` coercion below)
/// is what proves the umbrella exports the very trait the concrete crate defines.
fn assert_queue<T: MessageQueue>() {}

#[test]
fn runner_paths_resolve() {
    assert_same_type::<Runner, openjiuwen_runner::Runner>();
    assert_same_type::<RunOptions, openjiuwen_runner::RunOptions>();
    assert_same_type::<ResourceMgr, openjiuwen_runner::ResourceMgr>();
    assert_same_type::<CallbackEvent, openjiuwen_runner::CallbackEvent>();
    assert_same_type::<InMemoryMessageQueue, openjiuwen_runner::InMemoryMessageQueue>();

    // `MessageQueue` is a trait, not a type: assert the concrete queue implements it
    // and that it still coerces to the trait object.
    assert_queue::<InMemoryMessageQueue>();
    assert_queue::<openjiuwen_runner::InMemoryMessageQueue>();
    // The constructors hand back an `Arc<Self>`, so coerce through it.
    let queue = InMemoryMessageQueue::with_capacity(4);
    let queue: &dyn MessageQueue = queue.as_ref();
    assert!(!queue.is_started(), "a queue is created stopped");
}

#[test]
fn controller_paths_resolve() {
    use openjiuwen_core::controller::{
        ControllerAgent, ControllerConfig, ControllerOutput, Event, Intent, LLMController, Task,
        TaskScheduler, WorkflowController,
    };

    assert_same_type::<ControllerAgent, openjiuwen_agent::controller::ControllerAgent>();
    assert_same_type::<ControllerConfig, openjiuwen_agent::controller::ControllerConfig>();
    assert_same_type::<Task, openjiuwen_agent::controller::Task>();
    assert_same_type::<Event, openjiuwen_agent::controller::Event>();
    assert_same_type::<Intent, openjiuwen_agent::controller::Intent>();
    assert_same_type::<TaskScheduler, openjiuwen_agent::controller::TaskScheduler>();
    assert_same_type::<LLMController, openjiuwen_agent::controller::LLMController>();
    assert_same_type::<WorkflowController, openjiuwen_agent::controller::WorkflowController>();
    assert_same_type::<ControllerOutput, openjiuwen_agent::controller::ControllerOutput>();

    // `TaskExecutor` is a trait (no `TypeId`): prove the concrete crate's trait
    // object satisfies the umbrella trait's bound.
    assert_task_executor::<dyn openjiuwen_agent::controller::TaskExecutor>();

    // Nested module paths stay reachable through the module glob.
    assert_same_type::<Task, openjiuwen_agent::controller::schema::Task>();
}

#[test]
fn kv_cache_paths_resolve() {
    use openjiuwen_core::kv_cache::{
        InProcessKVCacheRuntime, KVCacheBackend, KVCacheIdentity, KVCacheRuntimeConfig,
        KVCacheStats,
    };

    assert_same_type::<KVCacheRuntimeConfig, openjiuwen_agent::kv_cache::KVCacheRuntimeConfig>();
    assert_same_type::<KVCacheBackend, openjiuwen_agent::kv_cache::KVCacheBackend>();
    assert_same_type::<KVCacheIdentity, openjiuwen_agent::kv_cache::KVCacheIdentity>();
    assert_same_type::<KVCacheStats, openjiuwen_agent::kv_cache::KVCacheStats>();
    assert_same_type::<InProcessKVCacheRuntime, openjiuwen_agent::kv_cache::InProcessKVCacheRuntime>(
    );

    // Nested module path.
    assert_same_type::<
        InProcessKVCacheRuntime,
        openjiuwen_agent::kv_cache::runtime::InProcessKVCacheRuntime,
    >();
}

#[test]
fn operator_paths_resolve() {
    use openjiuwen_core::operator::{
        ContextEvolveOperator, LLMCallOperator, MemoryCallOperator, SkillCallOperator,
        ToolCallOperator, TunableSpec,
    };

    assert_same_type::<ToolCallOperator, openjiuwen_agent::operator::ToolCallOperator>();
    assert_same_type::<LLMCallOperator, openjiuwen_agent::operator::LLMCallOperator>();
    assert_same_type::<MemoryCallOperator, openjiuwen_agent::operator::MemoryCallOperator>();
    assert_same_type::<SkillCallOperator, openjiuwen_agent::operator::SkillCallOperator>();
    assert_same_type::<ContextEvolveOperator, openjiuwen_agent::operator::ContextEvolveOperator>();
    assert_same_type::<TunableSpec, openjiuwen_agent::operator::TunableSpec>();

    // Nested module path.
    assert_same_type::<TunableSpec, openjiuwen_agent::operator::base::TunableSpec>();
}

#[test]
fn single_agent_extended_paths_resolve() {
    use openjiuwen_core::single_agent::abilities::{
        load_mcp_abilities, parts::extract_parts, McpAbilityOptions, McpToolAbility,
        SubAgentAbility, WorkflowAbility,
    };
    use openjiuwen_core::single_agent::Observation;

    assert_same_type::<SubAgentAbility, openjiuwen_agent::abilities::SubAgentAbility>();
    assert_same_type::<WorkflowAbility, openjiuwen_agent::abilities::WorkflowAbility>();
    assert_same_type::<McpToolAbility, openjiuwen_agent::abilities::McpToolAbility>();
    assert_same_type::<McpAbilityOptions, openjiuwen_agent::abilities::McpAbilityOptions>();
    assert_same_type::<Observation, openjiuwen_agent::output::Observation>();

    // The multimodal `parts` field exists and is readable.
    let observation = Observation {
        parts: Vec::new(),
        ..Observation::default()
    };
    assert_eq!(observation.parts, Vec::new());

    // Function-item identity: both paths resolve to the same address.
    let core_extract: fn(&serde_json::Value) -> Vec<openjiuwen_foundation::ContentPart> =
        extract_parts;
    let agent_extract: fn(&serde_json::Value) -> Vec<openjiuwen_foundation::ContentPart> =
        openjiuwen_agent::abilities::parts::extract_parts;
    assert!(std::ptr::eq(
        core_extract as *const (),
        agent_extract as *const ()
    ));

    // Referenced, never called: the async MCP loader resolves through the alias.
    reference_item(load_mcp_abilities);

    assert_eq!(
        openjiuwen_core::single_agent::output::LLM_ROUTE,
        openjiuwen_agent::output::LLM_ROUTE
    );
}

/// Proves the `single_agent` glob forwards the agent crate's `memory_provider`,
/// `usage`, and `prompt_pack` submodules unchanged.
#[test]
fn single_agent_modules_paths_resolve() {
    use openjiuwen_core::single_agent::{
        memory_provider::{provider_tool_cards, provider_tools, ProviderMemoryRail, ProviderTool},
        prompt_pack::{
            directory::PromptDirectory, PromptPack, PromptPackContext, PromptSectionsRail,
            RailSpec, PROMPT_SECTIONS_METADATA_KEY,
        },
        usage::{default_usage_analyzer, UsageTelemetry, MAIN_CONTEXT_ID},
    };

    // The external memory-provider bridge: the rail, the tool wrapper, and the two
    // tool-schema builders.
    assert_same_type::<ProviderMemoryRail, openjiuwen_agent::memory_provider::ProviderMemoryRail>();
    assert_same_type::<ProviderTool, openjiuwen_agent::memory_provider::ProviderTool>();
    let core_tools: ProviderToolsFn = provider_tools;
    let agent_tools: ProviderToolsFn = openjiuwen_agent::memory_provider::provider_tools;
    assert!(std::ptr::eq(
        core_tools as *const (),
        agent_tools as *const ()
    ));
    let core_cards: fn(
        &std::sync::Arc<dyn openjiuwen_memory::MemoryProvider>,
    ) -> Vec<openjiuwen_foundation::ToolCard> = provider_tool_cards;
    let agent_cards: fn(
        &std::sync::Arc<dyn openjiuwen_memory::MemoryProvider>,
    ) -> Vec<openjiuwen_foundation::ToolCard> =
        openjiuwen_agent::memory_provider::provider_tool_cards;
    assert!(std::ptr::eq(
        core_cards as *const (),
        agent_cards as *const ()
    ));

    // Usage telemetry: the drained buffer, the fixed context id, and the default
    // analyzer a host gets without configuring anything.
    assert_same_type::<UsageTelemetry, openjiuwen_agent::usage::UsageTelemetry>();
    assert_eq!(MAIN_CONTEXT_ID, openjiuwen_agent::usage::MAIN_CONTEXT_ID);
    let core_analyzer: fn() -> openjiuwen_context::ContextUsageAnalyzer = default_usage_analyzer;
    let agent_analyzer: fn() -> openjiuwen_context::ContextUsageAnalyzer =
        openjiuwen_agent::usage::default_usage_analyzer;
    assert!(std::ptr::eq(
        core_analyzer as *const (),
        agent_analyzer as *const ()
    ));

    // YAML prompt packs and the sections rail they install.
    assert_same_type::<PromptPack, openjiuwen_agent::prompt_pack::PromptPack>();
    assert_same_type::<PromptPackContext, openjiuwen_agent::prompt_pack::PromptPackContext>();
    assert_same_type::<PromptSectionsRail, openjiuwen_agent::prompt_pack::PromptSectionsRail>();
    assert_same_type::<RailSpec, openjiuwen_agent::prompt_pack::RailSpec>();
    assert_eq!(
        PROMPT_SECTIONS_METADATA_KEY,
        openjiuwen_agent::prompt_pack::PROMPT_SECTIONS_METADATA_KEY
    );

    // Nested module paths.
    assert_same_type::<PromptDirectory, openjiuwen_agent::prompt_pack::directory::PromptDirectory>(
    );
    assert_same_type::<PromptPack, openjiuwen_agent::prompt_pack::pack::PromptPack>();
}

#[test]
fn session_extended_paths_resolve() {
    use openjiuwen_core::session::{
        diff_snapshots, install_trace_sink, DiskSessionStore, RedisCheckpointer, RouterSession,
        SessionController, SessionHistory, SessionScope, SubWorkflowSession, SwappableTraceSink,
        Version,
    };

    assert_same_type::<SessionController, openjiuwen_session::SessionController>();
    assert_same_type::<DiskSessionStore, openjiuwen_session::DiskSessionStore>();
    assert_same_type::<SessionScope, openjiuwen_session::SessionScope>();
    assert_same_type::<SessionHistory, openjiuwen_session::SessionHistory>();
    assert_same_type::<Version, openjiuwen_session::Version>();
    assert_same_type::<SubWorkflowSession, openjiuwen_session::SubWorkflowSession>();
    assert_same_type::<RouterSession, openjiuwen_session::RouterSession>();
    assert_same_type::<RedisCheckpointer, openjiuwen_session::RedisCheckpointer>();
    assert_same_type::<SwappableTraceSink, openjiuwen_session::SwappableTraceSink>();

    // Function items: bind both paths to the same fn-pointer type and address.
    let core_install: InstallTraceSinkFn = install_trace_sink;
    let concrete_install: InstallTraceSinkFn = openjiuwen_session::install_trace_sink;
    assert!(std::ptr::eq(
        core_install as *const (),
        concrete_install as *const ()
    ));

    let core_diff: DiffSnapshotsFn = diff_snapshots;
    let concrete_diff: DiffSnapshotsFn = openjiuwen_session::diff_snapshots;
    assert!(std::ptr::eq(
        core_diff as *const (),
        concrete_diff as *const ()
    ));
}

#[test]
fn runner_extended_paths_resolve() {
    use openjiuwen_core::runner::{
        AuthFilter, CallbackChain, HandlerBuilder, InteractionHub, McpServerMgr, ModelMgr, PoolKey,
        PoolLease, PromptMgr, RemoteAgent, RemoteAgentConfig, RunEvent, RunMonitor,
        SubprocessSpawner, SysOperationMgr, TagMgr, TeamPool, WrapRegistry,
    };

    // Crate-root surface.
    assert_same_type::<TeamPool, openjiuwen_runner::TeamPool>();
    assert_same_type::<PoolLease, openjiuwen_runner::PoolLease>();
    assert_same_type::<PoolKey, openjiuwen_runner::PoolKey>();
    assert_same_type::<TagMgr, openjiuwen_runner::TagMgr>();
    assert_same_type::<ModelMgr, openjiuwen_runner::ModelMgr>();
    assert_same_type::<PromptMgr, openjiuwen_runner::PromptMgr>();
    assert_same_type::<SysOperationMgr, openjiuwen_runner::SysOperationMgr>();
    assert_same_type::<McpServerMgr, openjiuwen_runner::McpServerMgr>();
    assert_same_type::<InteractionHub, openjiuwen_runner::InteractionHub>();
    assert_same_type::<RunMonitor, openjiuwen_runner::RunMonitor>();
    assert_same_type::<RunEvent, openjiuwen_runner::RunEvent>();
    assert_same_type::<SubprocessSpawner, openjiuwen_runner::SubprocessSpawner>();
    assert_same_type::<RemoteAgent, openjiuwen_runner::RemoteAgent>();
    assert_same_type::<RemoteAgentConfig, openjiuwen_runner::RemoteAgentConfig>();
    assert_same_type::<CallbackChain, openjiuwen_runner::CallbackChain>();
    assert_same_type::<AuthFilter, openjiuwen_runner::AuthFilter>();
    assert_same_type::<HandlerBuilder, openjiuwen_runner::HandlerBuilder>();
    assert_same_type::<WrapRegistry, openjiuwen_runner::WrapRegistry>();

    // Nested module paths reach the same items.
    assert_same_type::<TeamPool, openjiuwen_runner::pool::TeamPool>();
    assert_same_type::<PoolLease, openjiuwen_runner::pool::PoolLease>();
    assert_same_type::<PoolKey, openjiuwen_runner::pool::PoolKey>();
    assert_same_type::<TagMgr, openjiuwen_runner::resources::TagMgr>();
    assert_same_type::<CallbackChain, openjiuwen_runner::callback::chain::CallbackChain>();
    assert_same_type::<InteractionHub, openjiuwen_runner::hitl::InteractionHub>();
    assert_same_type::<RunMonitor, openjiuwen_runner::monitor::RunMonitor>();
    assert_same_type::<RunEvent, openjiuwen_runner::monitor::RunEvent>();
    assert_same_type::<SubprocessSpawner, openjiuwen_runner::spawn::SubprocessSpawner>();
}

#[test]
fn foundation_extended_paths_resolve() {
    use openjiuwen_core::foundation::llm::{
        oauth::OAuthToken, reasoning::ReasoningConfig, routing::RoutingModel, AnthropicClient,
    };
    use openjiuwen_core::foundation::store::{ChromaVectorStore, StoreRegistry};
    use openjiuwen_core::foundation::tool::mcp::McpTool;

    assert_same_type::<StoreRegistry, openjiuwen_foundation::store::StoreRegistry>();
    assert_same_type::<ChromaVectorStore, openjiuwen_foundation::store::ChromaVectorStore>();
    assert_same_type::<AnthropicClient, openjiuwen_foundation::llm::AnthropicClient>();
    assert_same_type::<ReasoningConfig, openjiuwen_foundation::llm::reasoning::ReasoningConfig>();
    assert_same_type::<RoutingModel, openjiuwen_foundation::llm::routing::RoutingModel>();
    assert_same_type::<OAuthToken, openjiuwen_foundation::llm::oauth::OAuthToken>();
    assert_same_type::<McpTool, openjiuwen_foundation::tool::mcp::McpTool>();
}

#[test]
fn memory_extended_paths_resolve() {
    use openjiuwen_core::memory::{
        analyzer::MemoryAnalyzer,
        dreaming::DreamJob,
        external::ProviderRegistry,
        lite::{
            local::LocalCodingMemory,
            markdown::{FrontMatter, MarkdownMemory},
            CodingFact, CodingMemory,
        },
        team::TeamMemoryRecord,
    };

    // Nested module paths.
    assert_same_type::<CodingMemory, openjiuwen_memory::lite::CodingMemory>();
    assert_same_type::<MemoryAnalyzer, openjiuwen_memory::analyzer::MemoryAnalyzer>();
    assert_same_type::<ProviderRegistry, openjiuwen_memory::external::ProviderRegistry>();
    assert_same_type::<TeamMemoryRecord, openjiuwen_memory::team::TeamMemoryRecord>();
    assert_same_type::<DreamJob, openjiuwen_memory::dreaming::DreamJob>();

    // The lite coding-memory stack: typed facts, the Markdown source of truth, and
    // the local file-backed store. (`lite::fts` is sqlite-gated; it is proved in
    // `tests/feature_forwarding.rs`.)
    assert_same_type::<CodingFact, openjiuwen_memory::lite::CodingFact>();
    assert_same_type::<MarkdownMemory, openjiuwen_memory::lite::markdown::MarkdownMemory>();
    assert_same_type::<FrontMatter, openjiuwen_memory::lite::markdown::FrontMatter>();
    assert_same_type::<LocalCodingMemory, openjiuwen_memory::lite::local::LocalCodingMemory>();

    // Crate-root re-exports of the same items.
    assert_same_type::<CodingMemory, openjiuwen_memory::CodingMemory>();
    assert_same_type::<MemoryAnalyzer, openjiuwen_memory::MemoryAnalyzer>();
    assert_same_type::<DreamJob, openjiuwen_memory::DreamJob>();
}

/// Proves the `memory` alias forwards the memory crate's `lite` extensions,
/// `dreaming`, `team`, `external`, and `store::migration` submodules unchanged.
#[test]
fn memory_modules_paths_resolve() {
    use openjiuwen_core::memory::{
        dreaming::{DreamConfig, DreamOrchestrator, DreamState},
        external::{ProviderRegistry, RestProvider},
        lite::{
            embed::{cosine_similarity, EmbeddingCache, EmbeddingIndex, MemChunk},
            facts::{slug, FactDocument, FactFiles, MEMORY_FILE},
            local::LocalCodingMemoryBuilder,
            session::{sanitize_session_id, SessionStore},
        },
        store::migration::{standard_registry, MessageMigrator, SqlMigrator, VectorMigrator},
        team::{TeamMemoryHooks, TeamMemoryRecord},
    };

    // Lite coding-memory extensions: per-fact files, the embedding index, session
    // logs, and the local-store builder.
    assert_same_type::<FactFiles, openjiuwen_memory::lite::facts::FactFiles>();
    assert_same_type::<FactDocument, openjiuwen_memory::lite::facts::FactDocument>();
    assert_eq!(MEMORY_FILE, openjiuwen_memory::lite::facts::MEMORY_FILE);
    let core_slug: fn(&str) -> String = slug;
    let concrete_slug: fn(&str) -> String = openjiuwen_memory::lite::facts::slug;
    assert!(std::ptr::eq(
        core_slug as *const (),
        concrete_slug as *const ()
    ));

    assert_same_type::<EmbeddingCache, openjiuwen_memory::lite::embed::EmbeddingCache>();
    assert_same_type::<MemChunk, openjiuwen_memory::lite::embed::MemChunk>();
    assert_same_type::<EmbeddingIndex, openjiuwen_memory::lite::embed::EmbeddingIndex>();
    let core_cosine: fn(&[f32], &[f32]) -> f32 = cosine_similarity;
    let concrete_cosine: fn(&[f32], &[f32]) -> f32 =
        openjiuwen_memory::lite::embed::cosine_similarity;
    assert!(std::ptr::eq(
        core_cosine as *const (),
        concrete_cosine as *const ()
    ));

    assert_same_type::<SessionStore, openjiuwen_memory::lite::session::SessionStore>();
    let core_sanitize: fn(&str) -> openjiuwen_common::error::Result<String> = sanitize_session_id;
    let concrete_sanitize: fn(&str) -> openjiuwen_common::error::Result<String> =
        openjiuwen_memory::lite::session::sanitize_session_id;
    assert!(std::ptr::eq(
        core_sanitize as *const (),
        concrete_sanitize as *const ()
    ));

    assert_same_type::<
        LocalCodingMemoryBuilder,
        openjiuwen_memory::lite::local::LocalCodingMemoryBuilder,
    >();

    // The dream-job queue and its orchestrator.
    assert_same_type::<DreamConfig, openjiuwen_memory::dreaming::DreamConfig>();
    assert_same_type::<DreamState, openjiuwen_memory::dreaming::DreamState>();
    assert_same_type::<DreamOrchestrator, openjiuwen_memory::dreaming::DreamOrchestrator>();

    // Team-scoped memory: the record, the hooks, and the store seam. `TeamMemoryStore`
    // is a trait, so the bound on the concrete trait object proves the alias.
    assert_same_type::<TeamMemoryRecord, openjiuwen_memory::team::TeamMemoryRecord>();
    assert_same_type::<TeamMemoryHooks, openjiuwen_memory::team::TeamMemoryHooks>();
    assert_team_memory_store::<dyn openjiuwen_memory::team::TeamMemoryStore>();

    // The external provider SPI: registry, REST transport, and the provider seam.
    assert_same_type::<ProviderRegistry, openjiuwen_memory::external::ProviderRegistry>();
    assert_same_type::<RestProvider, openjiuwen_memory::external::RestProvider>();
    assert_memory_provider::<dyn openjiuwen_memory::external::MemoryProvider>();

    // The forward-only migration runner. `store::migration::runner` is sqlite-gated, so
    // only the always-compiled migrators and the standard registry are named here.
    assert_same_type::<SqlMigrator, openjiuwen_memory::store::migration::SqlMigrator>();
    assert_same_type::<VectorMigrator, openjiuwen_memory::store::migration::VectorMigrator>();
    assert_same_type::<MessageMigrator, openjiuwen_memory::store::migration::MessageMigrator>();
    let core_registry: fn() -> openjiuwen_common::error::Result<
        openjiuwen_memory::store::migration::OperationRegistry,
    > = standard_registry;
    let concrete_registry: fn() -> openjiuwen_common::error::Result<
        openjiuwen_memory::store::migration::OperationRegistry,
    > = openjiuwen_memory::store::migration::standard_registry;
    assert!(std::ptr::eq(
        core_registry as *const (),
        concrete_registry as *const ()
    ));
}

#[test]
fn retrieval_extended_paths_resolve() {
    use openjiuwen_core::retrieval::{
        graph::{InMemoryGraphStore, Triple},
        multimodal::MultimodalDocument,
        parsing::PdfParser,
        rewrite::RewriteRequest,
        sparse::{Bm25Index, SparseVector},
    };

    assert_same_type::<SparseVector, openjiuwen_retrieval::sparse::SparseVector>();
    assert_same_type::<Bm25Index, openjiuwen_retrieval::sparse::Bm25Index>();
    assert_same_type::<RewriteRequest, openjiuwen_retrieval::rewrite::RewriteRequest>();
    assert_same_type::<Triple, openjiuwen_retrieval::graph::Triple>();
    assert_same_type::<InMemoryGraphStore, openjiuwen_retrieval::graph::InMemoryGraphStore>();
    assert_same_type::<MultimodalDocument, openjiuwen_retrieval::multimodal::MultimodalDocument>();
    assert_same_type::<PdfParser, openjiuwen_retrieval::parsing::PdfParser>();

    // The sparse/BM25 surface is reachable through its nested module as well.
    assert_same_type::<Bm25Index, openjiuwen_retrieval::sparse::bm25::Bm25Index>();
    assert_eq!(
        openjiuwen_core::retrieval::sparse::bm25::DEFAULT_STOPWORDS,
        openjiuwen_retrieval::sparse::bm25::DEFAULT_STOPWORDS
    );

    // `QueryRewriter` is a trait: prove the concrete trait object satisfies the
    // umbrella trait's bound.
    assert_query_rewriter::<dyn openjiuwen_retrieval::rewrite::QueryRewriter>();
}

/// Proves the `retrieval` alias forwards the loaders for the OOXML/ODF containers and
/// the pure-Rust containers and readers beneath them (`parsing::zip`, `pptx`, `odt`).
#[test]
fn retrieval_modules_paths_resolve() {
    use openjiuwen_core::retrieval::{
        loaders::{
            fixtures::{build_zip, odt_document, pptx_slides},
            odt::OdtLoader,
            pptx::PptxLoader,
        },
        parsing::{
            odt::{OdtDocument, OdtParser},
            pptx::{PptxDocument, PptxParser, PptxSlide},
            zip::{ZipArchive, ZipEntry, ZipWriter},
        },
    };

    // The two office loaders.
    assert_same_type::<PptxLoader, openjiuwen_retrieval::loaders::pptx::PptxLoader>();
    assert_same_type::<OdtLoader, openjiuwen_retrieval::loaders::odt::OdtLoader>();
    assert_same_type::<PptxLoader, openjiuwen_retrieval::loaders::PptxLoader>();
    assert_same_type::<OdtLoader, openjiuwen_retrieval::loaders::OdtLoader>();

    // The in-memory document builders. Binding them is I/O-free; they are never called.
    let core_build_zip: BuildZipFn = build_zip;
    let concrete_build_zip: BuildZipFn = openjiuwen_retrieval::loaders::fixtures::build_zip;
    assert!(std::ptr::eq(
        core_build_zip as *const (),
        concrete_build_zip as *const ()
    ));

    let core_pptx: PptxSlidesFn = pptx_slides;
    let concrete_pptx: PptxSlidesFn = openjiuwen_retrieval::loaders::fixtures::pptx_slides;
    assert!(std::ptr::eq(
        core_pptx as *const (),
        concrete_pptx as *const ()
    ));

    let core_odt: fn(&[&str]) -> openjiuwen_common::error::Result<Vec<u8>> = odt_document;
    let concrete_odt: fn(&[&str]) -> openjiuwen_common::error::Result<Vec<u8>> =
        openjiuwen_retrieval::loaders::fixtures::odt_document;
    assert!(std::ptr::eq(
        core_odt as *const (),
        concrete_odt as *const ()
    ));

    // The containers under the parsers.
    assert_same_type::<ZipWriter, openjiuwen_retrieval::parsing::zip::ZipWriter>();
    assert_same_type::<ZipArchive, openjiuwen_retrieval::parsing::zip::ZipArchive>();
    assert_same_type::<ZipEntry, openjiuwen_retrieval::parsing::zip::ZipEntry>();
    assert_same_type::<PptxParser, openjiuwen_retrieval::parsing::pptx::PptxParser>();
    assert_same_type::<PptxDocument, openjiuwen_retrieval::parsing::pptx::PptxDocument>();
    assert_same_type::<PptxSlide, openjiuwen_retrieval::parsing::pptx::PptxSlide>();
    assert_same_type::<OdtParser, openjiuwen_retrieval::parsing::odt::OdtParser>();
    assert_same_type::<OdtDocument, openjiuwen_retrieval::parsing::odt::OdtDocument>();

    // The flat `parsing` re-exports are the very same types.
    assert_same_type::<ZipWriter, openjiuwen_retrieval::parsing::ZipWriter>();
    assert_same_type::<ZipArchive, openjiuwen_retrieval::parsing::ZipArchive>();
    assert_same_type::<PptxParser, openjiuwen_retrieval::parsing::PptxParser>();
    assert_same_type::<OdtParser, openjiuwen_retrieval::parsing::OdtParser>();
}

#[test]
fn workflow_components_paths_resolve() {
    use openjiuwen_core::workflow::{
        BranchComponent, HttpRequestComponent, IntentDetectionComponent,
        KnowledgeRetrievalComponent, LlmComponent, LoopComponent, MemoryRetrievalComponent,
        MemoryWriteComponent, QuestionerComponent, ReActAgentComponent, SubWorkflowComponent,
        ToolComponent,
    };

    // Crate-root re-exports through the umbrella alias.
    assert_same_type::<SubWorkflowComponent, openjiuwen_workflow::SubWorkflowComponent>();
    assert_same_type::<QuestionerComponent, openjiuwen_workflow::QuestionerComponent>();
    assert_same_type::<IntentDetectionComponent, openjiuwen_workflow::IntentDetectionComponent>();
    assert_same_type::<ReActAgentComponent, openjiuwen_workflow::ReActAgentComponent>();
    assert_same_type::<MemoryRetrievalComponent, openjiuwen_workflow::MemoryRetrievalComponent>();
    assert_same_type::<MemoryWriteComponent, openjiuwen_workflow::MemoryWriteComponent>();
    assert_same_type::<KnowledgeRetrievalComponent, openjiuwen_workflow::KnowledgeRetrievalComponent>(
    );
    assert_same_type::<LlmComponent, openjiuwen_workflow::LlmComponent>();
    assert_same_type::<BranchComponent, openjiuwen_workflow::BranchComponent>();
    assert_same_type::<LoopComponent, openjiuwen_workflow::LoopComponent>();
    assert_same_type::<ToolComponent, openjiuwen_workflow::ToolComponent>();
    assert_same_type::<HttpRequestComponent, openjiuwen_workflow::HttpRequestComponent>();

    // Nested component module paths stay reachable through the umbrella.
    assert_same_type::<
        SubWorkflowComponent,
        openjiuwen_workflow::components::sub_workflow::SubWorkflowComponent,
    >();
    assert_same_type::<
        QuestionerComponent,
        openjiuwen_workflow::components::questioner::QuestionerComponent,
    >();
    assert_same_type::<
        HttpRequestComponent,
        openjiuwen_workflow::components::http_request::HttpRequestComponent,
    >();
}

/// Proves the `workflow` alias forwards the single-component runner (`single`), its
/// crate-root re-exports, and the thinner parts of the component configuration tree.
#[test]
fn workflow_single_component_paths_resolve() {
    use openjiuwen_core::workflow::{
        components::{
            http_request::HttpComponentConfig as NestedHttpComponentConfig,
            questioner::{
                FieldInfo as NestedFieldInfo, QuestionerConfig as NestedQuestionerConfig,
            },
        },
        single::{execute_single_component, execute_single_component_with, SingleComponentOptions},
        FieldInfo, HttpComponentConfig, QuestionerConfig, SUB_WORKFLOW_COMPONENT,
    };

    // The single-component options type and its crate-root re-export.
    assert_same_type::<SingleComponentOptions, openjiuwen_workflow::single::SingleComponentOptions>(
    );
    assert_same_type::<SingleComponentOptions, openjiuwen_workflow::SingleComponentOptions>();

    // Both runners are `async fn`s, so their fn-item types are unnameable: reference
    // them through each path instead of binding a fn pointer. Never called.
    reference_item(execute_single_component);
    reference_item(openjiuwen_workflow::single::execute_single_component);
    reference_item(openjiuwen_workflow::execute_single_component);
    reference_item(execute_single_component_with);
    reference_item(openjiuwen_workflow::single::execute_single_component_with);
    reference_item(openjiuwen_workflow::execute_single_component_with);

    // Questioner configuration, through the umbrella root and its owning module.
    assert_same_type::<
        QuestionerConfig,
        openjiuwen_workflow::components::questioner::QuestionerConfig,
    >();
    assert_same_type::<FieldInfo, openjiuwen_workflow::components::questioner::FieldInfo>();
    assert_same_type::<NestedQuestionerConfig, openjiuwen_workflow::QuestionerConfig>();
    assert_same_type::<NestedFieldInfo, openjiuwen_workflow::FieldInfo>();

    // HTTP-request configuration, through the umbrella root and its owning module.
    assert_same_type::<
        HttpComponentConfig,
        openjiuwen_workflow::components::http_request::HttpComponentConfig,
    >();
    assert_same_type::<NestedHttpComponentConfig, openjiuwen_workflow::HttpComponentConfig>();

    // The sub-workflow component's registry key is the same constant at both paths.
    assert_eq!(
        SUB_WORKFLOW_COMPONENT,
        openjiuwen_workflow::SUB_WORKFLOW_COMPONENT
    );
}
