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

#[test]
fn core_alias_points_at_the_kernel_crates() {
    same_type::<openjiuwen::core::common::error::Error, openjiuwen_common::error::Error>();
    same_type::<openjiuwen::core::workflow::Workflow, openjiuwen_workflow::Workflow>();
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
