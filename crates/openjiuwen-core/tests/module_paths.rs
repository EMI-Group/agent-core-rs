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
