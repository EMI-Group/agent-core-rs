//! The default LLM-driven supervisor of a hierarchical team.
//!
//! Python parity: `SupervisorAgent` in
//! `openjiuwen/core/multi_agent/teams/hierarchical_msgbus/supervisor_agent.py`.
//! Python builds it by multiple inheritance (`CommunicableAgent` + `ReActAgent`)
//! and injects a `P2PAbilityManager` that turns sub-agent cards into LLM tools.
//! Rust composes the same pieces: [`SupervisorAgent::create`] returns a
//! [`ReActAgent`] plus a bind hook that mounts one
//! [`AgentTool`](crate::AgentTool) per sub-agent card the first time the member
//! is resolved by the team runtime.
//!
//! The provider builds a **fresh** agent per resolution (Python parity), and
//! each agent gets its own tool semaphore, so two concurrent team invocations
//! never share a parallel-dispatch budget.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use openjiuwen_agent::{Agent, AgentCard, ReActAgent, ReActAgentConfig};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::prompt::BaseMessage;
use openjiuwen_foundation::tool::Tool;
use tokio::sync::Semaphore;

use crate::agent_tool::AgentTool;
use crate::member::{AgentProvider, TeamAgent, ToolHost};
use crate::runtime::TeamRuntime;

/// Default number of `ReAct` iterations of the supervisor loop.
const DEFAULT_MAX_ITERATIONS: usize = 5;
/// Default number of concurrent sub-agent dispatches.
const DEFAULT_MAX_PARALLEL_SUB_AGENTS: usize = 10;

/// Configuration of the default supervisor.
///
/// Process-local by design: the model is a trait object and the system prompt is
/// Rust-owned text, so the type is neither serializable nor schema-visible.
#[derive(Clone)]
pub struct SupervisorConfig {
    /// Card of the supervisor agent.
    pub card: AgentCard,
    /// Model driving the supervisor's `ReAct` loop.
    pub model: Arc<dyn Model>,
    /// System prompt prepended to every supervisor run.
    pub system_prompt: String,
    /// Maximum `ReAct` iterations of one supervisor run.
    pub max_iterations: usize,
    /// Maximum number of sub-agent dispatches that may run concurrently.
    pub max_parallel_sub_agents: usize,
}

impl std::fmt::Debug for SupervisorConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SupervisorConfig")
            .field("card", &self.card.id)
            .field("model", &self.model.model_name())
            .field("system_prompt", &self.system_prompt)
            .field("max_iterations", &self.max_iterations)
            .field("max_parallel_sub_agents", &self.max_parallel_sub_agents)
            .finish()
    }
}

impl SupervisorConfig {
    /// Builds a configuration with the Python defaults (`5` iterations, `10`
    /// parallel sub-agents).
    #[must_use]
    pub fn new(card: AgentCard, model: Arc<dyn Model>, system_prompt: impl Into<String>) -> Self {
        Self {
            card,
            model,
            system_prompt: system_prompt.into(),
            max_iterations: DEFAULT_MAX_ITERATIONS,
            max_parallel_sub_agents: DEFAULT_MAX_PARALLEL_SUB_AGENTS,
        }
    }
}

/// The default supervisor factory.
///
/// A unit struct because every piece of state lives in the provider closure;
/// it exists so the Python class-level entry point keeps its shape.
#[derive(Debug, Clone, Copy, Default)]
pub struct SupervisorAgent;

impl SupervisorAgent {
    /// Builds the supervisor card and the provider that resolves it.
    ///
    /// The provider constructs a fresh [`ReActAgent`] per resolution, configures
    /// the model and system prompt, and mounts one
    /// [`AgentTool`](crate::AgentTool) per card of `agents` — bounded by one
    /// shared [`Semaphore`] — the first time the runtime binds the instance.
    ///
    /// The tool set is fixed here: later team members are not visible to the
    /// supervisor (Python passes this list up front too).
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::ErrorKind::Validation`] when `agents` is
    /// empty, or when a card has a blank id or name, or an id or name that
    /// duplicates an earlier card.
    pub fn create(
        agents: Vec<AgentCard>,
        config: SupervisorConfig,
    ) -> Result<(AgentCard, AgentProvider)> {
        validate_agents(&agents)?;

        let SupervisorConfig {
            card,
            model,
            system_prompt,
            max_iterations,
            max_parallel_sub_agents,
        } = config;
        let provider_card = card.clone();
        let parallel_dispatches = max_parallel_sub_agents.max(1);

        let provider: AgentProvider = Arc::new(move || {
            let agent = Arc::new(ReActAgent::new(provider_card.clone()));
            agent.configure(ReActAgentConfig {
                max_iterations,
                prompt_template: vec![BaseMessage::system(system_prompt.clone())],
                ..ReActAgentConfig::default()
            });
            agent.set_model(Arc::clone(&model));

            let limiter = Arc::new(Semaphore::new(parallel_dispatches));
            let hook_agent = Arc::clone(&agent);
            let hook_cards = agents.clone();
            let hook_limiter = Arc::clone(&limiter);
            let mounted = Arc::new(AtomicBool::new(false));
            let bind = move |runtime: &Arc<TeamRuntime>, _agent_id: &str| {
                if mounted.swap(true, Ordering::SeqCst) {
                    return;
                }
                for sub_card in &hook_cards {
                    let tool: Arc<dyn Tool> = Arc::new(
                        AgentTool::new(sub_card, Arc::downgrade(runtime))
                            .with_limiter(Arc::clone(&hook_limiter)),
                    );
                    let _ = hook_agent.add_tool(tool);
                }
            };

            let agent_handle: Arc<dyn Agent> = agent.clone();
            let tool_host: Arc<dyn ToolHost> = agent.clone();
            TeamAgent::with_tool_host(agent_handle, tool_host).with_bind(bind)
        });

        Ok((card, provider))
    }
}

/// Builds the supervisor card and provider.
///
/// Convenience free function delegating to [`SupervisorAgent::create`].
///
/// # Errors
///
/// See [`SupervisorAgent::create`].
pub fn create_supervisor(
    agents: Vec<AgentCard>,
    config: SupervisorConfig,
) -> Result<(AgentCard, AgentProvider)> {
    SupervisorAgent::create(agents, config)
}

/// Rejects an empty sub-agent list, blank identities, and duplicates.
fn validate_agents(agents: &[AgentCard]) -> Result<()> {
    if agents.is_empty() {
        return Err(Error::validation(
            "[SupervisorAgent.create] agents list must not be empty",
        ));
    }
    let mut ids: BTreeSet<&str> = BTreeSet::new();
    let mut names: BTreeSet<&str> = BTreeSet::new();
    for card in agents {
        if card.id.trim().is_empty() {
            return Err(Error::validation(format!(
                "[SupervisorAgent.create] agent card '{}' has a blank agent id",
                card.name
            )));
        }
        if card.name.trim().is_empty() {
            return Err(Error::validation(format!(
                "[SupervisorAgent.create] agent card '{}' has a blank agent name",
                card.id
            )));
        }
        if !ids.insert(card.id.as_str()) {
            return Err(Error::validation(format!(
                "[SupervisorAgent.create] duplicate agent id '{}'",
                card.id
            )));
        }
        if !names.insert(card.name.as_str()) {
            return Err(Error::validation(format!(
                "[SupervisorAgent.create] duplicate agent name '{}'",
                card.name
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_foundation::llm::Model;
    use openjiuwen_test_support::ScriptedModel;

    use super::*;
    use crate::runtime::RuntimeConfig;

    fn card(id: &str) -> AgentCard {
        AgentCard::new(id).with_id(id)
    }

    fn config() -> SupervisorConfig {
        let model: Arc<dyn Model> = Arc::new(ScriptedModel::text("ok"));
        SupervisorConfig::new(card("boss"), model, "you are the boss")
    }

    #[test]
    fn new_fills_the_python_defaults() {
        let config = config();
        assert_eq!(config.max_iterations, 5);
        assert_eq!(config.max_parallel_sub_agents, 10);
        assert_eq!(config.system_prompt, "you are the boss");
        assert_eq!(config.card.id, "boss");
    }

    #[test]
    fn create_rejects_an_empty_agent_list() {
        let error = SupervisorAgent::create(vec![], config())
            .err()
            .expect("a validation error");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(
            error.message(),
            "[SupervisorAgent.create] agents list must not be empty"
        );
    }

    #[test]
    fn create_rejects_blank_identities() {
        let blank_name = AgentCard::new("").with_id("worker");
        let error = SupervisorAgent::create(vec![blank_name], config())
            .err()
            .expect("a validation error");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.message().contains("blank agent name"));

        let blank_id = AgentCard::new("worker").with_id("");
        let error = SupervisorAgent::create(vec![blank_id], config())
            .err()
            .expect("a validation error");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.message().contains("blank agent id"));
    }

    #[test]
    fn create_rejects_duplicates() {
        let error = SupervisorAgent::create(vec![card("worker"), card("worker")], config())
            .err()
            .expect("a validation error");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(
            error.message(),
            "[SupervisorAgent.create] duplicate agent id 'worker'"
        );

        let renamed = AgentCard::new("worker").with_id("other");
        let error = SupervisorAgent::create(vec![card("worker"), renamed], config())
            .err()
            .expect("a validation error");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(
            error.message(),
            "[SupervisorAgent.create] duplicate agent name 'worker'"
        );
    }

    #[test]
    fn create_returns_the_supervisor_card() {
        let config = config();
        let (card, _provider) = create_supervisor(vec![card("worker")], config.clone()).unwrap();
        assert_eq!(card, config.card);
        let wire = serde_json::to_value(&card).unwrap();
        let reparsed: AgentCard = serde_json::from_value(wire).unwrap();
        assert_eq!(reparsed, card);
    }

    #[test]
    fn provider_builds_a_fresh_tool_hosting_member() {
        let (_card, provider) = SupervisorAgent::create(vec![card("worker")], config()).unwrap();
        let first = provider();
        let second = provider();
        assert_eq!(first.agent().card().id, "boss");
        assert!(first.tool_host().is_some());
        assert!(
            !Arc::ptr_eq(&first.agent(), &second.agent()),
            "the provider must build a fresh agent per resolution"
        );
    }

    #[test]
    fn binding_only_mounts_tools_and_registers_nothing() {
        let (_card, provider) =
            SupervisorAgent::create(vec![card("worker"), card("researcher")], config()).unwrap();
        let member = provider();
        let runtime = TeamRuntime::new(RuntimeConfig::new());
        member.bind(&runtime, "boss");
        assert_eq!(runtime.agent_count(), 0);
        // Binding the same instance again must not mount the tools twice; the
        // provider-built member silently ignores the repeated bind.
        member.bind(&runtime, "boss");
        assert_eq!(runtime.agent_count(), 0);
    }

    #[test]
    fn zero_parallel_sub_agents_still_builds_a_provider() {
        let mut config = config();
        config.max_parallel_sub_agents = 0;
        let (_card, provider) = SupervisorAgent::create(vec![card("worker")], config).unwrap();
        let member = provider();
        assert!(member.tool_host().is_some());
        let runtime = TeamRuntime::new(RuntimeConfig::new());
        member.bind(&runtime, "boss");
        assert_eq!(runtime.agent_count(), 0);
    }
}
