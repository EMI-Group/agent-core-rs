//! The runner's registry of published resources: tools, agents, workflows, and
//! agent teams.
//!
//! Ports Python `openjiuwen/core/runner/resources_manager/resource_manager.py`
//! reduced to the four public resource kinds hosts depend on. Each kind owns one
//! [`Registry`] (see `registry`) keyed on the resource card id, with the
//! provider invoked on every lookup (Python parity: a fresh instance per
//! `get_*`).
//!
//! Tools are the exception to the "fresh instance" shape: a host registers a
//! live `Arc<dyn Tool>`, so the built-in provider hands out a clone of that same
//! `Arc`.
//!
//! Not ported (see the crate-level `## Deferred Python Behavior`): tags
//! (`TagMgr`), MCP servers/tools, sys-operations, models, and prompts.

mod registry;

use std::sync::Arc;

use openjiuwen_agent::{Agent, AgentCard};
use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::{Tool, ToolCard};
use openjiuwen_multiagent::{Team, TeamCard};
use openjiuwen_workflow::{Workflow, WorkflowCard};

pub use registry::{Provider, RegisterPolicy, Registration, Registry, ResourceCard};

/// A factory for tools. Tools are registered from a live instance, so the
/// provider returns a clone of the registered `Arc`.
pub type ToolProvider = Provider<Arc<dyn Tool>>;

/// A factory for agents; invoked once per lookup, so it may build per-run state.
pub type AgentProvider = Provider<Arc<dyn Agent>>;

/// A factory for workflows; invoked once per lookup.
pub type WorkflowProvider = Provider<Workflow>;

/// A factory for agent teams; invoked once per lookup.
pub type TeamProvider = Provider<Arc<dyn Team>>;

/// The four typed registries the runner exposes to a host.
///
/// The manager is shared as `Arc<ResourceMgr>` by the owning
/// [`Runner`](crate::Runner); every method takes `&self` and mutates through the
/// per-registry lock. Each registry keeps its own revision counter, so a host
/// caching a projection of one kind is invalidated only by changes to that kind.
///
/// Duplicate ids follow the caller's [`RegisterPolicy`]; the default
/// `register_*` methods use [`RegisterPolicy::Fail`], which keeps the first
/// registration and returns
/// [`openjiuwen_common::error::Error::Validation`].
#[derive(Debug)]
pub struct ResourceMgr {
    /// Registered tools, keyed on [`ToolCard::id`].
    tools: Registry<ToolCard, Arc<dyn Tool>>,
    /// Registered agents, keyed on [`AgentCard::id`].
    agents: Registry<AgentCard, Arc<dyn Agent>>,
    /// Registered workflows, keyed on [`WorkflowCard::id`].
    workflows: Registry<WorkflowCard, Workflow>,
    /// Registered teams, keyed on [`TeamCard::id`].
    teams: Registry<TeamCard, Arc<dyn Team>>,
}

impl ResourceMgr {
    /// Builds an empty manager with all four registries.
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            tools: Registry::with_kind("tool"),
            agents: Registry::with_kind("agent"),
            workflows: Registry::with_kind("workflow"),
            teams: Registry::with_kind("team"),
        })
    }

    // ---------------------------------------------------------------- tools --

    /// Registers `tool`, failing when its card id is already present.
    ///
    /// The card is copied from the tool and the provider returns the same `Arc`
    /// on every lookup, so the registered instance is the one handed out.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Validation`] for an empty card id or a
    /// duplicate registration.
    pub fn register_tool(&self, tool: Arc<dyn Tool>) -> Result<()> {
        self.register_tool_with(tool, RegisterPolicy::Fail)
            .map(|_| ())
    }

    /// Registers `tool` under `policy`, reporting whether an entry was replaced.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Validation`] for an empty card id and,
    /// under [`RegisterPolicy::Fail`], for a duplicate registration.
    pub fn register_tool_with(
        &self,
        tool: Arc<dyn Tool>,
        policy: RegisterPolicy,
    ) -> Result<Registration> {
        let card = tool.card().clone();
        let provider: ToolProvider = Arc::new(move || Arc::clone(&tool));
        self.tools.register(card, provider, policy)
    }

    /// Returns the tool registered under `id`, or `None`.
    ///
    /// The provider runs on every call, so a missing id invokes nothing.
    #[must_use]
    pub fn tool(&self, id: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(id)
    }

    /// Returns the tool whose card name is `name` (first registration wins).
    #[must_use]
    pub fn tool_by_name(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get_by_name(name)
    }

    /// Returns a clone of the card registered under `id`.
    #[must_use]
    pub fn tool_card(&self, id: &str) -> Option<ToolCard> {
        self.tools.card(id)
    }

    /// Removes `id` and returns its card.
    #[must_use]
    pub fn remove_tool(&self, id: &str) -> Option<ToolCard> {
        self.tools.remove(id)
    }

    /// Returns clones of the registered tool cards in registration order.
    #[must_use]
    pub fn tools(&self) -> Vec<ToolCard> {
        self.tools.cards()
    }

    /// Returns the registered tool ids in registration order.
    #[must_use]
    pub fn tool_ids(&self) -> Vec<String> {
        self.tools.ids()
    }

    /// Returns the tool registry's mutation counter.
    #[must_use]
    pub fn tool_revision(&self) -> u64 {
        self.tools.revision()
    }

    /// Removes every tool and returns how many were removed.
    pub fn clear_tools(&self) -> usize {
        self.tools.clear()
    }

    /// Returns the number of registered tools.
    #[must_use]
    pub fn tool_count(&self) -> usize {
        self.tools.len()
    }

    // --------------------------------------------------------------- agents --

    /// Registers `card` with `provider`, failing when the id is already present.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Validation`] for an empty card id or a
    /// duplicate registration.
    pub fn register_agent(&self, card: AgentCard, provider: AgentProvider) -> Result<()> {
        self.register_agent_with(card, provider, RegisterPolicy::Fail)
            .map(|_| ())
    }

    /// Registers `card` with `provider` under `policy`.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Validation`] for an empty card id and,
    /// under [`RegisterPolicy::Fail`], for a duplicate registration.
    pub fn register_agent_with(
        &self,
        card: AgentCard,
        provider: AgentProvider,
        policy: RegisterPolicy,
    ) -> Result<Registration> {
        self.agents.register(card, provider, policy)
    }

    /// Returns the agent registered under `id`, or `None`.
    #[must_use]
    pub fn agent(&self, id: &str) -> Option<Arc<dyn Agent>> {
        self.agents.get(id)
    }

    /// Returns the agent whose card name is `name` (first registration wins).
    #[must_use]
    pub fn agent_by_name(&self, name: &str) -> Option<Arc<dyn Agent>> {
        self.agents.get_by_name(name)
    }

    /// Returns a clone of the card registered under `id`.
    #[must_use]
    pub fn agent_card(&self, id: &str) -> Option<AgentCard> {
        self.agents.card(id)
    }

    /// Removes `id` and returns its card.
    #[must_use]
    pub fn remove_agent(&self, id: &str) -> Option<AgentCard> {
        self.agents.remove(id)
    }

    /// Returns clones of the registered agent cards in registration order.
    #[must_use]
    pub fn agents(&self) -> Vec<AgentCard> {
        self.agents.cards()
    }

    /// Returns the registered agent ids in registration order.
    #[must_use]
    pub fn agent_ids(&self) -> Vec<String> {
        self.agents.ids()
    }

    /// Returns the agent registry's mutation counter.
    #[must_use]
    pub fn agent_revision(&self) -> u64 {
        self.agents.revision()
    }

    /// Removes every agent and returns how many were removed.
    pub fn clear_agents(&self) -> usize {
        self.agents.clear()
    }

    /// Returns the number of registered agents.
    #[must_use]
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }

    // ------------------------------------------------------------ workflows --

    /// Registers `card` with `provider`, failing when the id is already present.
    ///
    /// Python versions a workflow by composing its id with
    /// [`workflow_key`]; the Rust registry keys on the card id exactly as the
    /// caller set it.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Validation`] for an empty card id or a
    /// duplicate registration.
    pub fn register_workflow(&self, card: WorkflowCard, provider: WorkflowProvider) -> Result<()> {
        self.register_workflow_with(card, provider, RegisterPolicy::Fail)
            .map(|_| ())
    }

    /// Registers `card` with `provider` under `policy`.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Validation`] for an empty card id and,
    /// under [`RegisterPolicy::Fail`], for a duplicate registration.
    pub fn register_workflow_with(
        &self,
        card: WorkflowCard,
        provider: WorkflowProvider,
        policy: RegisterPolicy,
    ) -> Result<Registration> {
        self.workflows.register(card, provider, policy)
    }

    /// Returns the workflow registered under `id`, or `None`.
    #[must_use]
    pub fn workflow(&self, id: &str) -> Option<Workflow> {
        self.workflows.get(id)
    }

    /// Returns the workflow whose card name is `name` (first registration wins).
    #[must_use]
    pub fn workflow_by_name(&self, name: &str) -> Option<Workflow> {
        self.workflows.get_by_name(name)
    }

    /// Returns a clone of the card registered under `id`.
    #[must_use]
    pub fn workflow_card(&self, id: &str) -> Option<WorkflowCard> {
        self.workflows.card(id)
    }

    /// Removes `id` and returns its card.
    #[must_use]
    pub fn remove_workflow(&self, id: &str) -> Option<WorkflowCard> {
        self.workflows.remove(id)
    }

    /// Returns clones of the registered workflow cards in registration order.
    #[must_use]
    pub fn workflows(&self) -> Vec<WorkflowCard> {
        self.workflows.cards()
    }

    /// Returns the registered workflow ids in registration order.
    #[must_use]
    pub fn workflow_ids(&self) -> Vec<String> {
        self.workflows.ids()
    }

    /// Returns the workflow registry's mutation counter.
    #[must_use]
    pub fn workflow_revision(&self) -> u64 {
        self.workflows.revision()
    }

    /// Removes every workflow and returns how many were removed.
    pub fn clear_workflows(&self) -> usize {
        self.workflows.clear()
    }

    /// Returns the number of registered workflows.
    #[must_use]
    pub fn workflow_count(&self) -> usize {
        self.workflows.len()
    }

    // ---------------------------------------------------------------- teams --

    /// Registers `card` with `provider`, failing when the id is already present.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Validation`] for an empty card id or a
    /// duplicate registration.
    pub fn register_team(&self, card: TeamCard, provider: TeamProvider) -> Result<()> {
        self.register_team_with(card, provider, RegisterPolicy::Fail)
            .map(|_| ())
    }

    /// Registers `card` with `provider` under `policy`.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Validation`] for an empty card id and,
    /// under [`RegisterPolicy::Fail`], for a duplicate registration.
    pub fn register_team_with(
        &self,
        card: TeamCard,
        provider: TeamProvider,
        policy: RegisterPolicy,
    ) -> Result<Registration> {
        self.teams.register(card, provider, policy)
    }

    /// Returns the team registered under `id`, or `None`.
    #[must_use]
    pub fn team(&self, id: &str) -> Option<Arc<dyn Team>> {
        self.teams.get(id)
    }

    /// Returns the team whose card name is `name` (first registration wins).
    #[must_use]
    pub fn team_by_name(&self, name: &str) -> Option<Arc<dyn Team>> {
        self.teams.get_by_name(name)
    }

    /// Returns a clone of the card registered under `id`.
    #[must_use]
    pub fn team_card(&self, id: &str) -> Option<TeamCard> {
        self.teams.card(id)
    }

    /// Removes `id` and returns its card.
    #[must_use]
    pub fn remove_team(&self, id: &str) -> Option<TeamCard> {
        self.teams.remove(id)
    }

    /// Returns clones of the registered team cards in registration order.
    #[must_use]
    pub fn teams(&self) -> Vec<TeamCard> {
        self.teams.cards()
    }

    /// Returns the registered team ids in registration order.
    #[must_use]
    pub fn team_ids(&self) -> Vec<String> {
        self.teams.ids()
    }

    /// Returns the team registry's mutation counter.
    #[must_use]
    pub fn team_revision(&self) -> u64 {
        self.teams.revision()
    }

    /// Removes every team and returns how many were removed.
    pub fn clear_teams(&self) -> usize {
        self.teams.clear()
    }

    /// Returns the number of registered teams.
    #[must_use]
    pub fn team_count(&self) -> usize {
        self.teams.len()
    }

    // ------------------------------------------------------------- lifecycle --

    /// Returns whether all four registries are empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
            && self.agents.is_empty()
            && self.workflows.is_empty()
            && self.teams.is_empty()
    }

    /// Clears all four registries and resets every revision counter to zero.
    ///
    /// Python parity: `Runner::stop` calls `ResourceMgr.release`, returning the
    /// process to a pristine registry state.
    pub fn release(&self) {
        self.tools.reset();
        self.agents.reset();
        self.workflows.reset();
        self.teams.reset();
    }
}

/// Composes a workflow registry key from its id and version.
///
/// Python parity: `generate_workflow_key` (and the `ResourceMgr` workflow
/// manager) stores workflows under `"{id}_{version}"`.
#[must_use]
pub fn workflow_key(id: &str, version: &str) -> String {
    format!("{id}_{version}")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use futures::future::BoxFuture;
    use futures::stream::BoxStream;
    use openjiuwen_agent::{AgentChunk, AgentOutput};
    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_foundation::tool::{FunctionTool, ToolContext, ToolHandler};
    use openjiuwen_multiagent::{handoff_team, HandoffConfig, HandoffTeamConfig};
    use openjiuwen_session::Session;
    use serde_json::{json, Value};

    /// An agent that answers with its own card id; enough to prove the seam.
    struct FakeAgent {
        card: AgentCard,
    }

    #[async_trait::async_trait]
    impl Agent for FakeAgent {
        fn card(&self) -> &AgentCard {
            &self.card
        }

        async fn invoke(
            &self,
            _inputs: Value,
            _session: Option<Arc<dyn Session>>,
        ) -> Result<AgentOutput> {
            Ok(AgentOutput {
                output: format!("answer:{}", self.card.id),
                ..AgentOutput::default()
            })
        }

        fn stream(
            &self,
            _inputs: Value,
            _session: Option<Arc<dyn Session>>,
        ) -> BoxStream<'static, Result<AgentChunk>> {
            Box::pin(futures::stream::empty())
        }
    }

    /// An agent provider minting a fresh [`FakeAgent`] per lookup.
    fn agent_provider(card: AgentCard) -> AgentProvider {
        Arc::new(move || Arc::new(FakeAgent { card: card.clone() }) as Arc<dyn Agent>)
    }

    /// A workflow provider minting a fresh (empty) workflow per lookup.
    fn workflow_provider(card: WorkflowCard) -> WorkflowProvider {
        Arc::new(move || Workflow::new(card.clone()))
    }

    /// A team provider minting a fresh strategy-free team per lookup.
    fn team_provider(card: TeamCard) -> TeamProvider {
        Arc::new(move || {
            let config = HandoffTeamConfig::new(HandoffConfig::default());
            Arc::new(handoff_team(card.clone(), config)) as Arc<dyn Team>
        })
    }

    /// An echoing tool whose card carries `card`.
    fn echo_tool(card: ToolCard) -> Arc<dyn Tool> {
        let handler: ToolHandler = Arc::new(
            |inputs, _ctx: ToolContext| -> BoxFuture<'static, Result<Value>> {
                Box::pin(async move { Ok(inputs) })
            },
        );
        FunctionTool::new(card, handler).boxed()
    }

    fn agent_card(id: &str, name: &str) -> AgentCard {
        AgentCard::default().with_id(id).with_name(name)
    }

    fn tool_card(id: &str, name: &str) -> ToolCard {
        ToolCard::new(name, "echoes its inputs", json!({})).with_id(id)
    }

    fn workflow_card(id: &str, name: &str) -> WorkflowCard {
        WorkflowCard {
            id: id.to_owned(),
            name: name.to_owned(),
            ..WorkflowCard::default()
        }
    }

    fn team_card(id: &str, name: &str) -> TeamCard {
        TeamCard::new(name).with_id(id)
    }

    // ports tests/unit_tests/core/runner/test_resource_manager.py
    #[test]
    fn each_kind_registers_looks_up_and_removes() {
        let manager = ResourceMgr::new();
        assert!(manager.is_empty());

        manager
            .register_tool(echo_tool(tool_card("t1", "echo")))
            .unwrap();
        manager
            .register_agent(
                agent_card("a1", "assistant"),
                agent_provider(agent_card("a1", "assistant")),
            )
            .unwrap();
        manager
            .register_workflow(
                workflow_card("w1", "flow"),
                workflow_provider(workflow_card("w1", "flow")),
            )
            .unwrap();
        manager
            .register_team(
                team_card("g1", "crew"),
                team_provider(team_card("g1", "crew")),
            )
            .unwrap();

        assert_eq!(manager.tool_count(), 1);
        assert_eq!(manager.agent_count(), 1);
        assert_eq!(manager.workflow_count(), 1);
        assert_eq!(manager.team_count(), 1);
        assert!(!manager.is_empty());

        assert_eq!(manager.tool_card("t1").unwrap().name, "echo");
        assert_eq!(manager.tool_by_name("echo").unwrap().card().id, "t1");
        assert_eq!(manager.agent_card("a1").unwrap().name, "assistant");
        assert_eq!(manager.agent_by_name("assistant").unwrap().card().id, "a1");
        assert_eq!(manager.workflow_card("w1").unwrap().name, "flow");
        assert_eq!(manager.workflow_by_name("flow").unwrap().card().id, "w1");
        assert_eq!(manager.team_card("g1").unwrap().name, "crew");
        assert_eq!(manager.team_by_name("crew").unwrap().card().id, "g1");

        assert_eq!(manager.remove_tool("t1").unwrap().id, "t1");
        assert_eq!(manager.remove_agent("a1").unwrap().id, "a1");
        assert_eq!(manager.remove_workflow("w1").unwrap().id, "w1");
        assert_eq!(manager.remove_team("g1").unwrap().id, "g1");
        assert!(manager.is_empty());
        assert!(manager.remove_tool("t1").is_none());
        assert_eq!(manager.tool_revision(), 2);
    }

    // ports tests/unit_tests/core/runner/test_resource_manager.py
    #[test]
    fn empty_ids_are_rejected_with_the_typed_noun() {
        let manager = ResourceMgr::new();

        let error = manager
            .register_tool(echo_tool(tool_card("  ", "ghost")))
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(
            error.to_string(),
            "validation: tool id cannot be empty or whitespace only"
        );

        let error = manager
            .register_agent(
                agent_card("", "ghost"),
                agent_provider(agent_card("a", "a")),
            )
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "validation: agent id cannot be empty or whitespace only"
        );

        let error = manager
            .register_workflow(
                workflow_card(" ", "ghost"),
                workflow_provider(workflow_card("w", "w")),
            )
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "validation: workflow id cannot be empty or whitespace only"
        );

        let error = manager
            .register_team(team_card("", "ghost"), team_provider(team_card("g", "g")))
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "validation: team id cannot be empty or whitespace only"
        );

        assert!(manager.is_empty());
    }

    // ports tests/unit_tests/core/runner/test_resource_manager.py
    #[test]
    fn duplicate_agents_fail_by_default() {
        let manager = ResourceMgr::new();
        manager
            .register_agent(
                agent_card("a1", "assistant"),
                agent_provider(agent_card("a1", "assistant")),
            )
            .unwrap();

        let error = manager
            .register_agent(
                agent_card("a1", "assistant"),
                agent_provider(agent_card("a1", "assistant")),
            )
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(
            error.to_string(),
            "validation: agent 'a1' is already registered"
        );
        assert_eq!(manager.agent_count(), 1);
        assert_eq!(manager.agent_revision(), 1);
    }

    #[test]
    fn release_clears_every_registry_and_resets_revisions() {
        let manager = ResourceMgr::new();
        manager
            .register_tool(echo_tool(tool_card("t1", "echo")))
            .unwrap();
        manager
            .register_agent(
                agent_card("a1", "assistant"),
                agent_provider(agent_card("a1", "assistant")),
            )
            .unwrap();
        manager
            .register_workflow(
                workflow_card("w1", "flow"),
                workflow_provider(workflow_card("w1", "flow")),
            )
            .unwrap();
        manager
            .register_team(
                team_card("g1", "crew"),
                team_provider(team_card("g1", "crew")),
            )
            .unwrap();

        manager.release();

        assert!(manager.is_empty());
        assert_eq!(manager.tool_revision(), 0);
        assert_eq!(manager.agent_revision(), 0);
        assert_eq!(manager.workflow_revision(), 0);
        assert_eq!(manager.team_revision(), 0);
    }

    #[test]
    fn clear_tools_reports_the_count_and_bumps_the_revision() {
        let manager = ResourceMgr::new();
        manager
            .register_tool(echo_tool(tool_card("t1", "echo")))
            .unwrap();
        manager
            .register_tool(echo_tool(tool_card("t2", "echo2")))
            .unwrap();

        assert_eq!(manager.clear_tools(), 2);
        assert_eq!(manager.tool_count(), 0);
        assert_eq!(manager.tool_revision(), 3);
        assert_eq!(manager.clear_tools(), 0);
    }

    #[test]
    fn workflow_key_matches_python() {
        assert_eq!(workflow_key("wf", "1.0.0"), "wf_1.0.0");
        assert_eq!(workflow_key("wf", ""), "wf_");
        assert_eq!(workflow_key("a_b", "v2"), "a_b_v2");
    }
}
