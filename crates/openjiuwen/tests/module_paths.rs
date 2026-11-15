//! Identity checks for the umbrella facade.
//!
//! Every alias is compared against the concrete (canonical) item it forwards, so a
//! mis-pointed alias — or an alias that silently shadows a different type — fails here.
//! Type identity is asserted with [`std::any::TypeId`]; function identity with
//! [`std::ptr::eq`] on coerced function pointers.

use std::any::TypeId;

use openjiuwen_common::error::Result;
use openjiuwen_harness_protocol::{HarnessProtocol, JsonObject};

/// Asserts that `A` and `B` are the same type.
fn same_type<A: 'static, B: 'static>() {
    assert_eq!(TypeId::of::<A>(), TypeId::of::<B>());
}

/// Asserts that each umbrella path is the same type as the concrete item after `=>`.
macro_rules! same_paths {
    ($($umbrella:ty => $concrete:ty),+ $(,)?) => {
        $(same_type::<$umbrella, $concrete>();)+
    };
}

#[test]
fn core_alias_points_at_the_kernel_crates() {
    same_type::<openjiuwen::core::common::error::Error, openjiuwen_common::error::Error>();
    same_type::<openjiuwen::core::workflow::Workflow, openjiuwen_workflow::Workflow>();
}

#[test]
fn core_alias_exposes_the_new_kernel_modules() {
    // `controller`, `kv_cache` and `operator` are Python top-level packages hosted
    // by `openjiuwen-agent`; the kernel umbrella globs those modules wholesale.
    same_type::<
        openjiuwen::core::controller::ControllerAgent,
        openjiuwen_agent::controller::ControllerAgent,
    >();
    same_type::<
        openjiuwen::core::kv_cache::KVCacheRuntimeConfig,
        openjiuwen_agent::kv_cache::KVCacheRuntimeConfig,
    >();
    same_type::<
        openjiuwen::core::operator::ToolCallOperator,
        openjiuwen_agent::operator::ToolCallOperator,
    >();
    // Nested spelling: the glob keeps the submodules reachable.
    same_type::<
        openjiuwen::core::controller::schema::Task,
        openjiuwen_agent::controller::schema::Task,
    >();
}

#[test]
fn every_public_module_tree_is_reachable() {
    // Sessions: disk-backed product sessions and local version history.
    same_paths! {
        openjiuwen::core::session::session_controller::DiskSessionStore => openjiuwen_session::session_controller::DiskSessionStore,
        openjiuwen::core::session::vcs::Version => openjiuwen_session::vcs::Version,
    }

    // Runner: pool, resources, HITL, monitoring, subprocess spawning.
    same_paths! {
        openjiuwen::core::runner::pool::PoolLease => openjiuwen_runner::pool::PoolLease,
        openjiuwen::core::runner::resources::TagMgr => openjiuwen_runner::resources::TagMgr,
        openjiuwen::core::runner::hitl::InteractionHub => openjiuwen_runner::hitl::InteractionHub,
        openjiuwen::core::runner::monitor::RunMonitor => openjiuwen_runner::monitor::RunMonitor,
        openjiuwen::core::runner::spawn::SubprocessSpawner => openjiuwen_runner::spawn::SubprocessSpawner,
    }

    // Foundation: the external-store registry.
    same_paths! {
        openjiuwen::core::foundation::store::external::StoreRegistry => openjiuwen_foundation::store::external::StoreRegistry,
    }

    // Memory: lite coding memory, analysis, external providers, team memory, dreaming.
    same_paths! {
        openjiuwen::core::memory::lite::CodingMemory => openjiuwen_memory::lite::CodingMemory,
        openjiuwen::core::memory::analyzer::MemoryAnalyzer => openjiuwen_memory::analyzer::MemoryAnalyzer,
        openjiuwen::core::memory::external::ProviderRegistry => openjiuwen_memory::external::ProviderRegistry,
        openjiuwen::core::memory::team::TeamMemoryRecord => openjiuwen_memory::team::TeamMemoryRecord,
        openjiuwen::core::memory::dreaming::DreamJob => openjiuwen_memory::dreaming::DreamJob,
    }

    // Retrieval: sparse and query-rewriting trees.
    same_paths! {
        openjiuwen::core::retrieval::sparse::Bm25Index => openjiuwen_retrieval::sparse::Bm25Index,
        openjiuwen::core::retrieval::rewrite::RewriteRequest => openjiuwen_retrieval::rewrite::RewriteRequest,
        openjiuwen::core::retrieval::parsing::PdfParser => openjiuwen_retrieval::parsing::PdfParser,
    }

    // Harness: LSP, spec registry, worktrees, subagents, built-in tool catalog.
    same_paths! {
        openjiuwen::harness::lsp::LspClient => openjiuwen_harness::lsp::LspClient,
        openjiuwen::harness::spec::SpecRegistry => openjiuwen_harness::spec::SpecRegistry,
        openjiuwen::harness::worktree::WorktreeManager => openjiuwen_harness::worktree::WorktreeManager,
        openjiuwen::harness::subagents::SubAgentSpec => openjiuwen_harness::subagents::SubAgentSpec,
        openjiuwen::harness::tools::BuiltinTool => openjiuwen_harness::tools::BuiltinTool,
    }

    // Agent teams: org namespacing is function-only, so identity is asserted on the
    // function pointer; the rest carry types.
    let org_umbrella: fn(&str) -> bool = openjiuwen::agent_teams::org::is_valid_org_id;
    let org_concrete: fn(&str) -> bool = openjiuwen_agent_teams::org::is_valid_org_id;
    assert!(std::ptr::eq(
        org_umbrella as *const (),
        org_concrete as *const ()
    ));

    same_paths! {
        openjiuwen::agent_teams::attach::AttachClient => openjiuwen_agent_teams::attach::client::AttachClient,
        openjiuwen::agent_teams::kv_cache::MemberCacheKey => openjiuwen_agent_teams::kv_cache::MemberCacheKey,
        openjiuwen::agent_teams::models::ModelPool => openjiuwen_agent_teams::models::ModelPool,
        openjiuwen::agent_teams::monitor::TeamMonitor => openjiuwen_agent_teams::monitor::TeamMonitor,
        openjiuwen::agent_teams::observability::TeamSpan => openjiuwen_agent_teams::observability::TeamSpan,
    }

    // Symphony: discovery, experience, flow, LLM judges, artifact persistence.
    same_paths! {
        openjiuwen::symphony::discovery::DiscoverySettings => openjiuwen_symphony::discovery::DiscoverySettings,
        openjiuwen::symphony::experience::ExperienceBank => openjiuwen_symphony::experience::ExperienceBank,
        openjiuwen::symphony::flow::FlowEngine => openjiuwen_symphony::flow::FlowEngine,
        openjiuwen::symphony::llm::LlmJudgeMetric => openjiuwen_symphony::llm::LlmJudgeMetric,
        openjiuwen::symphony::object_artifact::ObjectGraphArtifactStore => openjiuwen_symphony::object_artifact::ObjectGraphArtifactStore,
    }

    // Dev tools: judge, tune, skill creation, agent builder.
    same_paths! {
        openjiuwen::dev_tools::judge::ModelJudge => openjiuwen_dev_tools::judge::ModelJudge,
        openjiuwen::dev_tools::tune::Case => openjiuwen_dev_tools::tune::Case,
        openjiuwen::dev_tools::skill_creator::SkillCreator => openjiuwen_dev_tools::skill_creator::SkillCreator,
        openjiuwen::dev_tools::agent_builder::AgentBuilder => openjiuwen_dev_tools::agent_builder::AgentBuilder,
    }

    // Extensions: the context-evolution pipeline.
    same_paths! {
        openjiuwen::extensions::context_evolver::InMemoryMemoryStore => openjiuwen_extensions::context_evolver::InMemoryMemoryStore,
    }
}

#[test]
fn harness_alias_points_at_openjiuwen_harness() {
    same_type::<openjiuwen::harness::DeepAgent, openjiuwen_harness::agent::DeepAgent>();
}

#[test]
fn harness_protocol_alias_points_at_openjiuwen_harness_protocol() {
    same_type::<
        openjiuwen::harness_protocol::HarnessState,
        openjiuwen_harness_protocol::state::HarnessState,
    >();
}

#[test]
fn harness_providers_alias_points_at_openjiuwen_harness_providers() {
    same_type::<
        openjiuwen::harness_providers::ProviderDescriptor,
        openjiuwen_harness_providers::provider::ProviderDescriptor,
    >();

    let umbrella: fn(&str, JsonObject) -> Result<Box<dyn HarnessProtocol>> =
        openjiuwen::harness_providers::create_harness;
    let concrete: fn(&str, JsonObject) -> Result<Box<dyn HarnessProtocol>> =
        openjiuwen_harness_providers::factory::create_harness;
    assert!(std::ptr::eq(umbrella as *const (), concrete as *const ()));
}

#[test]
fn agent_teams_alias_points_at_openjiuwen_agent_teams() {
    same_type::<
        openjiuwen::agent_teams::TeamAgentSpec,
        openjiuwen_agent_teams::schema::TeamAgentSpec,
    >();
}

#[test]
fn extensions_alias_points_at_openjiuwen_extensions() {
    same_type::<
        openjiuwen::extensions::checkpointer::MemoryCheckpointer,
        openjiuwen_core::session::checkpoint::MemoryCheckpointer,
    >();
}

#[test]
fn dev_tools_alias_points_at_openjiuwen_dev_tools() {
    same_type::<
        openjiuwen::dev_tools::prompt_builder::Language,
        openjiuwen_dev_tools::prompt_builder::language::Language,
    >();
}

#[test]
fn symphony_alias_points_at_openjiuwen_symphony() {
    same_type::<
        openjiuwen::symphony::runtime::SymphonyRuntime,
        openjiuwen_symphony::runtime::SymphonyRuntime,
    >();
}

#[test]
fn rsi_alias_points_at_openjiuwen_rsi() {
    same_type::<openjiuwen::rsi::Candidate, openjiuwen_rsi::evolution::Candidate>();
    // The auto-harness orchestrator lives under the `harness_rsi` module in the concrete
    // crate; the umbrella forwards it under the shorter `rsi::auto_harness` path.
    same_paths! {
        openjiuwen::rsi::auto_harness::AutoHarnessConfig => openjiuwen_rsi::harness_rsi::auto_harness::AutoHarnessConfig,
        openjiuwen::rsi::auto_harness::AutoHarnessOrchestrator => openjiuwen_rsi::harness_rsi::auto_harness::AutoHarnessOrchestrator,
    }
}

#[test]
fn agent_evolving_alias_points_at_openjiuwen_agent_evolving() {
    same_type::<openjiuwen::agent_evolving::dataset::Case, openjiuwen_agent_evolving::dataset::Case>(
    );
}

#[test]
fn auto_harness_alias_points_at_openjiuwen_auto_harness() {
    // The shim crate exposes no types of its own, so identity is asserted on `crate_name`.
    let umbrella: fn() -> &'static str = openjiuwen::auto_harness::crate_name;
    let concrete: fn() -> &'static str = openjiuwen_auto_harness::crate_name;
    assert!(std::ptr::eq(umbrella as *const (), concrete as *const ()));
}

#[cfg(feature = "a2a")]
#[test]
fn a2a_is_reachable_through_the_extensions_alias() {
    same_type::<
        openjiuwen::extensions::a2a::client::A2aClient,
        openjiuwen_extensions::a2a::client::A2aClient,
    >();
}

#[cfg(feature = "otel")]
#[test]
fn otel_is_reachable_through_the_extensions_alias() {
    same_type::<
        openjiuwen::extensions::otel::OtelTracerConfig,
        openjiuwen_extensions::otel::OtelTracerConfig,
    >();
}

#[test]
fn core_alias_exposes_the_newest_agent_context_and_llm_trees() {
    // `single_agent` is a blanket glob of `openjiuwen-agent`, so the modules the crate
    // grew stay reachable under the Python spelling.
    same_paths! {
        openjiuwen::core::single_agent::memory_provider::ProviderMemoryRail => openjiuwen_agent::memory_provider::ProviderMemoryRail,
        openjiuwen::core::single_agent::usage::UsageTelemetry => openjiuwen_agent::usage::UsageTelemetry,
        openjiuwen::core::single_agent::controller::ControllerAgent => openjiuwen_agent::controller::ControllerAgent,
        openjiuwen::core::single_agent::kv_cache::KVCacheRuntimeConfig => openjiuwen_agent::kv_cache::KVCacheRuntimeConfig,
        openjiuwen::core::single_agent::operator::ToolCallOperator => openjiuwen_agent::operator::ToolCallOperator,
        openjiuwen::core::single_agent::prompt_pack::PromptPack => openjiuwen_agent::prompt_pack::PromptPack,
    }
    // Context engine: usage accounting, compression telemetry, window policy, tokenizers.
    same_paths! {
        openjiuwen::core::context_engine::usage::ContextUsageAnalyzer => openjiuwen_context::usage::ContextUsageAnalyzer,
        openjiuwen::core::context_engine::telemetry::ContextCompressionState => openjiuwen_context::telemetry::ContextCompressionState,
        openjiuwen::core::context_engine::policy::ContextWindowChange => openjiuwen_context::policy::ContextWindowChange,
        openjiuwen::core::context_engine::policy::ContextWindowPolicy => openjiuwen_context::policy::ContextWindowPolicy,
        openjiuwen::core::context_engine::tokenizers::TokenizerSpec => openjiuwen_context::tokenizers::TokenizerSpec,
    }
    // Foundation LLM: the external-provider registry and the `Responses` transport helpers.
    same_paths! {
        openjiuwen::core::foundation::llm::external_provider::ExternalProviderRegistry => openjiuwen_foundation::llm::external_provider::ExternalProviderRegistry,
        openjiuwen::core::foundation::llm::responses::ResponsesStreamState => openjiuwen_foundation::llm::responses::ResponsesStreamState,
    }
    // Runner and teams: the dreaming service and the team-memory configuration. (`core::memory`'s
    // `lite`/`dreaming`/`team`/`external` trees are already asserted in
    // `every_public_module_tree_is_reachable`.)
    same_paths! {
        openjiuwen::core::runner::dreaming::DreamService => openjiuwen_runner::dreaming::DreamService,
        openjiuwen::agent_teams::memory::MemoryConfig => openjiuwen_agent_teams::memory::MemoryConfig,
    }
}

#[test]
fn crate_name_is_the_python_package_root() {
    assert_eq!(openjiuwen::crate_name(), "openjiuwen");
}

#[test]
fn prelude_resolves_the_common_items() {
    use openjiuwen::prelude::*;

    // Proves the forwarded `Error`/`Result` pair is usable in signatures.
    fn not_found() -> Result<()> {
        Err(Error::not_found("prelude"))
    }

    assert!(not_found().is_err());

    let card = AgentCard::new("prelude-agent");
    assert_eq!(card.name, "prelude-agent");

    let message = BaseMessage::user("hi");
    assert_eq!(message.role, openjiuwen::core::foundation::Role::User);

    let session = AgentSession::new("s-1");
    assert_eq!(session.id(), "s-1");

    let team_spec = TeamAgentSpec::default();
    assert!(team_spec.agents.is_empty());
}

#[test]
fn prelude_resolves_the_agent_construction_items() {
    use openjiuwen::prelude::*;

    same_type::<ReActAgent, openjiuwen_agent::react::ReActAgent>();
    same_type::<ReActAgentConfig, openjiuwen_agent::react::ReActAgentConfig>();
    same_type::<AgentInput, openjiuwen_agent::input::AgentInput>();
    same_type::<AgentOutput, openjiuwen_agent::output::AgentOutput>();
    same_type::<AgentChunk, openjiuwen_agent::output::AgentChunk>();
    same_type::<ResultType, openjiuwen_agent::output::ResultType>();
    same_type::<LlmAgent, openjiuwen_agent::llm_agent::LlmAgent>();
    same_type::<WorkflowAgent, openjiuwen_agent::workflow_agent::WorkflowAgent>();
    same_type::<DeepAgentBuilder, openjiuwen_harness::agent::DeepAgentBuilder>();

    // The prelude keeps forwarding the lifecycle/context items.
    same_type::<ContextEngine, openjiuwen_context::engine::ContextEngine>();
    same_type::<LongTermMemory, openjiuwen_memory::facade::LongTermMemory>();
    same_type::<Runner, openjiuwen_runner::runner::Runner>();

    // A prelude-only usage path: both items are constructible without a model.
    let config = ReActAgentConfig::default();
    assert_eq!(config.model_provider, "openai");
    assert_eq!(config.max_iterations, 5);

    let input = AgentInput::default();
    assert!(input.conversation_id.is_none());
}

#[test]
fn prelude_resolves_the_extended_items() {
    use openjiuwen::prelude::*;

    // Controller stack, product sessions, and the runner's pool/monitor surface.
    same_type::<ControllerAgent, openjiuwen_agent::controller::ControllerAgent>();
    same_type::<SessionController, openjiuwen_session::session_controller::SessionController>();
    same_type::<TeamPool, openjiuwen_runner::pool::TeamPool>();
    same_type::<RunMonitor, openjiuwen_runner::monitor::RunMonitor>();
    same_type::<Observation, openjiuwen_agent::output::Observation>();

    // Cheap, network-free construction paths prove the forwarded names are usable.
    let _monitor = RunMonitor::new();
    let _pool = TeamPool::new();
    let observation = Observation::default();
    assert_eq!(observation.content, "");
}

#[cfg(feature = "otel")]
#[test]
fn otel_feature_reaches_the_kernel_trace_sink_hook() {
    // The umbrella's `otel` feature forwards to `openjiuwen-core/otel`, so the kernel
    // session's trace-sink hook is the session crate's installed-sink path. Referenced
    // as a function item only — installing a sink needs a live session.
    let umbrella: fn(
        &dyn openjiuwen_session::Session,
        std::sync::Arc<dyn openjiuwen_session::TraceSink>,
    ) -> Result<()> = openjiuwen::core::session::install_trace_sink;
    let concrete: fn(
        &dyn openjiuwen_session::Session,
        std::sync::Arc<dyn openjiuwen_session::TraceSink>,
    ) -> Result<()> = openjiuwen_session::install_trace_sink;
    assert!(std::ptr::eq(umbrella as *const (), concrete as *const ()));
}

#[cfg(feature = "otel")]
#[test]
fn otel_feature_reaches_the_agent_teams_span_sink() {
    // The umbrella's `otel` feature also forwards to `openjiuwen-agent-teams/otel`, which
    // compiles the team observability sink in.
    same_paths! {
        openjiuwen::agent_teams::observability::OtelSpanSink => openjiuwen_agent_teams::observability::OtelSpanSink,
        openjiuwen::agent_teams::observability::OtelHooks => openjiuwen_agent_teams::observability::OtelHooks,
        openjiuwen::agent_teams::observability::SpanRecord => openjiuwen_agent_teams::observability::SpanRecord,
    }
}

#[test]
fn crate_version_is_exported() {
    // Python's `openjiuwen/__init__.py` exports `__version__`; the facade mirrors it as `VERSION`.
    assert_eq!(openjiuwen::VERSION, env!("CARGO_PKG_VERSION"));
    assert_ne!(openjiuwen::VERSION, "");
}

#[cfg(feature = "native-v2")]
#[test]
fn native_v2_feature_reaches_the_agent_teams_hosted_provider() {
    same_paths! {
        openjiuwen::harness_providers::providers::native_v2::NativeV2Config => openjiuwen_harness_providers::providers::native_v2::NativeV2Config,
        openjiuwen::harness_providers::providers::native_v2::NativeV2HarnessProvider => openjiuwen_harness_providers::providers::native_v2::NativeV2HarnessProvider,
    }

    // With the forwarded feature the built-in table resolves the name instead of
    // answering `Unsupported`.
    let resolved = openjiuwen::harness_providers::providers::builtin_provider(
        openjiuwen_harness_providers::provider::provider_names::NATIVE_V2,
    );
    assert!(
        matches!(resolved, Some(Ok(_))),
        "`native-v2` must make the native_v2 provider resolvable through the umbrella"
    );
}

#[cfg(feature = "vendor")]
#[test]
fn vendor_feature_reaches_the_vendor_providers() {
    same_paths! {
        openjiuwen::harness_providers::providers::claudecode::ClaudeCodeHarnessConfig => openjiuwen_harness_providers::providers::claudecode::ClaudeCodeHarnessConfig,
        openjiuwen::harness_providers::providers::claudecode::ClaudeCodeHarnessProvider => openjiuwen_harness_providers::providers::claudecode::ClaudeCodeHarnessProvider,
        openjiuwen::harness_providers::providers::codex::CodexHarnessConfig => openjiuwen_harness_providers::providers::codex::CodexHarnessConfig,
        openjiuwen::harness_providers::providers::codex::CodexHarnessProvider => openjiuwen_harness_providers::providers::codex::CodexHarnessProvider,
        openjiuwen::harness_providers::providers::dsh::DshHarnessConfig => openjiuwen_harness_providers::providers::dsh::DshHarnessConfig,
        openjiuwen::harness_providers::providers::dsh::DshHarnessProvider => openjiuwen_harness_providers::providers::dsh::DshHarnessProvider,
    }

    // Registration only: driving a vendor provider needs the vendor CLI, so nothing here
    // constructs or runs one.
    for name in [
        openjiuwen_harness_providers::provider::provider_names::CLAUDE_CODE,
        openjiuwen_harness_providers::provider::provider_names::CODEX,
        openjiuwen_harness_providers::provider::provider_names::DSH,
    ] {
        let resolved = openjiuwen::harness_providers::providers::builtin_provider(name);
        assert!(
            matches!(resolved, Some(Ok(_))),
            "`vendor` must register `{name}` through the umbrella"
        );
    }
}

#[cfg(feature = "sqlite")]
#[test]
fn sqlite_feature_reaches_every_backend() {
    same_paths! {
        openjiuwen::core::session::checkpoint::SqliteCheckpointer => openjiuwen_session::checkpoint::SqliteCheckpointer,
        openjiuwen::core::foundation::store::SqliteKvStore => openjiuwen_foundation::store::SqliteKvStore,
        openjiuwen::extensions::store::SqliteKvStore => openjiuwen_core::foundation::store::SqliteKvStore,
        openjiuwen::agent_teams::storage::SqliteTeamStore => openjiuwen_agent_teams::storage::SqliteTeamStore,
    }

    // The feature-gated constructors are referenced by building their futures: an async fn
    // body does not run until it is polled, so no database is opened.
    let _memory_stores_from_umbrella =
        openjiuwen::core::memory::store::MemoryStores::sqlite("/nonexistent/memory");
    let _memory_stores_from_concrete =
        openjiuwen_memory::store::MemoryStores::sqlite("/nonexistent/memory");
    let _asset_store_from_umbrella =
        openjiuwen::symphony::store::KvAssetStore::sqlite("sqlite::memory:");
    let _asset_store_from_concrete =
        openjiuwen_symphony::store::KvAssetStore::sqlite("sqlite::memory:");
}

#[cfg(feature = "redis")]
#[test]
fn redis_feature_reaches_the_store_surface() {
    // Construction returns `Error::Unsupported`; nothing here connects.
    same_paths! {
        openjiuwen::extensions::store::redis::RedisStore => openjiuwen_extensions::store::redis::RedisStore,
        openjiuwen::extensions::store::redis::RedisStoreConfig => openjiuwen_extensions::store::redis::RedisStoreConfig,
    }
}

#[cfg(feature = "redis")]
#[test]
fn redis_feature_also_reaches_the_agent_evolving_store() {
    // The umbrella's `redis` feature forwards both halves: the extensions' store surface
    // and the RL online service's Redis-backed `OnlineStore`. Nothing here connects.
    same_paths! {
        openjiuwen::agent_evolving::online::redis_store::RedisOnlineStore => openjiuwen_agent_evolving::online::RedisOnlineStore,
        openjiuwen::agent_evolving::online::redis_store::RedisOnlineStoreOptions => openjiuwen_agent_evolving::online::RedisOnlineStoreOptions,
    }
}

#[cfg(feature = "otlp")]
#[test]
fn otlp_feature_reaches_the_team_memory_bridge() {
    // `otlp` is separate from `otel`: agent-teams keeps the in-memory/tracing seam and the
    // real OTLP memory bridge behind two features.
    same_paths! {
        openjiuwen::agent_teams::observability::otlp::OtlpMemoryObserver => openjiuwen_agent_teams::observability::otlp::OtlpMemoryObserver,
    }
    // The constructor takes `impl Into<String>`, so the coercion to a higher-ranked `fn(&str)`
    // is rejected; the identity check instead instantiates it at `String`, which stays a plain
    // function-pointer comparison of the two paths.
    let umbrella: fn(
        String,
    )
        -> Result<openjiuwen::agent_teams::observability::otlp::OtlpMemoryObserver> =
        openjiuwen::agent_teams::observability::otlp::otlp_memory_observer;
    let concrete: fn(
        String,
    )
        -> Result<openjiuwen_agent_teams::observability::otlp::OtlpMemoryObserver> =
        openjiuwen_agent_teams::observability::otlp::otlp_memory_observer;
    assert!(std::ptr::eq(umbrella as *const (), concrete as *const ()));
    // The crate-wide baseline is still "no exporter"; the constructor builds over a private
    // in-memory provider, so nothing touches the network.
    const { assert!(!openjiuwen::agent_teams::observability::OTLP_EXPORT_AVAILABLE) };
    let observer = openjiuwen::agent_teams::observability::otlp::otlp_memory_observer("s-1");
    assert!(
        observer.is_ok(),
        "the private in-memory memory provider builds"
    );
}

#[cfg(feature = "pulsar")]
#[test]
fn pulsar_feature_reaches_the_message_queue_surface() {
    // Construction returns `Error::Unsupported`; nothing here connects.
    same_paths! {
        openjiuwen::extensions::message_queue::PulsarConfig => openjiuwen_extensions::message_queue::PulsarConfig,
        openjiuwen::extensions::message_queue::PulsarMessageQueue => openjiuwen_extensions::message_queue::PulsarMessageQueue,
    }
}

#[cfg(feature = "mcp")]
#[test]
fn mcp_feature_is_wired_and_the_team_mcp_surface_resolves() {
    // The umbrella's `mcp` feature wires `openjiuwen-extensions/mcp`, which currently gates
    // no source; the runtime MCP server ships from `openjiuwen-agent-teams`, whose `mcp`
    // feature is on by default. This test keeps the umbrella feature exercised and proves
    // that surface resolves.
    same_paths! {
        openjiuwen::agent_teams::mcp::TeamMcpServer => openjiuwen_agent_teams::mcp::TeamMcpServer,
    }
    assert_eq!(
        openjiuwen::agent_teams::mcp::SERVER_NAME,
        openjiuwen_agent_teams::mcp::SERVER_NAME
    );
    assert_eq!(
        openjiuwen::agent_teams::mcp::OPERATOR_INSTRUCTIONS,
        openjiuwen_agent_teams::mcp::OPERATOR_INSTRUCTIONS
    );
}

#[cfg(feature = "distributed")]
#[test]
fn distributed_feature_reaches_the_messager_seam() {
    // The ZeroMQ messager is a construction-time-rejecting seam; only its type is asserted.
    same_paths! {
        openjiuwen::agent_teams::messager::PyZmqMessager => openjiuwen_agent_teams::messager::PyZmqMessager,
    }
}

#[cfg(feature = "sandbox")]
#[test]
fn sandbox_feature_reaches_the_provider_registry() {
    same_paths! {
        openjiuwen::core::sys_operation::sandbox::provider::SandboxProviderRegistry => openjiuwen_sysop::sandbox::provider::SandboxProviderRegistry,
    }
}

#[cfg(feature = "browser")]
#[test]
fn browser_feature_reaches_the_reporting_preset() {
    // The preset exists (the umbrella forwarded `openjiuwen-harness/browser`) but the agent
    // behind it stays deferred, exactly as the harness crate reports it.
    let result = openjiuwen::harness::subagents::build_preset(
        openjiuwen::harness::subagents::SubAgentPreset::Browser,
        "en",
    );
    let message = match result {
        Ok(_) => panic!("the browser preset agent is deferred"),
        Err(error) => error.to_string(),
    };
    assert!(
        message.contains("browser_agent") && message.contains("'browser' cargo feature"),
        "unexpected message: {message}"
    );
}

#[cfg(feature = "cua")]
#[test]
fn cua_feature_reaches_the_reporting_preset() {
    let result = openjiuwen::harness::subagents::build_preset(
        openjiuwen::harness::subagents::SubAgentPreset::Cua,
        "en",
    );
    let message = match result {
        Ok(_) => panic!("the cua preset agent is deferred"),
        Err(error) => error.to_string(),
    };
    assert!(
        message.contains("cua_agent") && message.contains("'cua' cargo feature"),
        "unexpected message: {message}"
    );
}

#[cfg(feature = "mobile")]
#[test]
fn mobile_feature_reaches_the_reporting_preset() {
    let result = openjiuwen::harness::subagents::build_preset(
        openjiuwen::harness::subagents::SubAgentPreset::Mobile,
        "en",
    );
    let message = match result {
        Ok(_) => panic!("the mobile preset agent is deferred"),
        Err(error) => error.to_string(),
    };
    assert!(
        message.contains("mobile_gui_agent") && message.contains("'mobile' cargo feature"),
        "unexpected message: {message}"
    );
}
