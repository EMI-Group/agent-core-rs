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
        analyzer::MemoryAnalyzer, dreaming::DreamJob, external::ProviderRegistry,
        lite::CodingMemory, team::TeamMemoryRecord,
    };

    // Nested module paths.
    assert_same_type::<CodingMemory, openjiuwen_memory::lite::CodingMemory>();
    assert_same_type::<MemoryAnalyzer, openjiuwen_memory::analyzer::MemoryAnalyzer>();
    assert_same_type::<ProviderRegistry, openjiuwen_memory::external::ProviderRegistry>();
    assert_same_type::<TeamMemoryRecord, openjiuwen_memory::team::TeamMemoryRecord>();
    assert_same_type::<DreamJob, openjiuwen_memory::dreaming::DreamJob>();

    // Crate-root re-exports of the same items.
    assert_same_type::<CodingMemory, openjiuwen_memory::CodingMemory>();
    assert_same_type::<MemoryAnalyzer, openjiuwen_memory::MemoryAnalyzer>();
    assert_same_type::<DreamJob, openjiuwen_memory::DreamJob>();
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

    // `QueryRewriter` is a trait: prove the concrete trait object satisfies the
    // umbrella trait's bound.
    assert_query_rewriter::<dyn openjiuwen_retrieval::rewrite::QueryRewriter>();
}
