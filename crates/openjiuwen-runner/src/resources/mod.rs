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
//! Alongside the four run-able registries the manager owns five sub-managers,
//! each reachable through an accessor: [`TagMgr`], [`ModelMgr`], [`PromptMgr`],
//! [`SysOperationMgr`], and [`McpServerMgr`].

pub mod mcp_server;
pub mod model;
pub mod prompt;
mod registry;
pub mod sysop;
pub mod tag;

use std::sync::Arc;

use openjiuwen_agent::{Agent, AgentCard};
use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::{Tool, ToolCard};
use openjiuwen_multiagent::{Team, TeamCard};
use openjiuwen_workflow::{Workflow, WorkflowCard};

pub use mcp_server::{
    DefaultMcpConnector, McpConnector, McpServerInfo, McpServerMgr, McpServerState,
};
pub use model::{ModelCard, ModelMgr, ModelProvider};
pub use prompt::{PromptCard, PromptMgr};
pub use registry::{Provider, RegisterPolicy, Registration, Registry, ResourceCard};
pub use sysop::{SysOperationMgr, SysOperationProvider};
pub use tag::{ResourceKind, Tag, TagMatchStrategy, TagMgr, TagUpdateStrategy, GLOBAL_TAG};

/// A factory for tools. Tools are registered from a live instance, so the
/// provider returns a clone of the registered `Arc`.
pub type ToolProvider = Provider<Arc<dyn Tool>>;

/// A factory for agents; invoked once per lookup, so it may build per-run state.
pub type AgentProvider = Provider<Arc<dyn Agent>>;

/// A factory for workflows; invoked once per lookup.
pub type WorkflowProvider = Provider<Workflow>;

/// A factory for agent teams; invoked once per lookup.
pub type TeamProvider = Provider<Arc<dyn Team>>;

/// The registries and sub-managers the runner exposes to a host.
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
pub struct ResourceMgr {
    /// Registered tools, keyed on [`ToolCard::id`].
    tools: Registry<ToolCard, Arc<dyn Tool>>,
    /// Registered agents, keyed on [`AgentCard::id`].
    agents: Registry<AgentCard, Arc<dyn Agent>>,
    /// Registered workflows, keyed on [`WorkflowCard::id`].
    workflows: Registry<WorkflowCard, Workflow>,
    /// Registered teams, keyed on [`TeamCard::id`].
    teams: Registry<TeamCard, Arc<dyn Team>>,
    /// The cross-kind tag index shared by all five resource kinds.
    tags: TagMgr,
    /// Model factories keyed on the model id.
    models: ModelMgr,
    /// Prompt templates keyed on the prompt id.
    prompts: PromptMgr,
    /// System operations keyed on the [`openjiuwen_sysop::SysOperationCard`] id.
    sysops: SysOperationMgr,
    /// MCP servers keyed on their `server_id`.
    mcp_servers: McpServerMgr,
}

impl ResourceMgr {
    /// Builds an empty manager with all four registries and five sub-managers.
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            tools: Registry::with_kind("tool"),
            agents: Registry::with_kind("agent"),
            workflows: Registry::with_kind("workflow"),
            teams: Registry::with_kind("team"),
            tags: TagMgr::new(),
            models: ModelMgr::new(),
            prompts: PromptMgr::new(),
            sysops: SysOperationMgr::new(),
            mcp_servers: McpServerMgr::new(),
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

    /// Removes `id`, returning its card and dropping the tool's tags.
    #[must_use]
    pub fn remove_tool(&self, id: &str) -> Option<ToolCard> {
        self.remove_tagged(&self.tools, ResourceKind::Tool, id)
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

    /// Removes every tool, dropping their tags, and returns how many were removed.
    pub fn clear_tools(&self) -> usize {
        self.clear_tagged(&self.tools, ResourceKind::Tool)
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

    /// Removes `id`, returning its card and dropping the agent's tags.
    #[must_use]
    pub fn remove_agent(&self, id: &str) -> Option<AgentCard> {
        self.remove_tagged(&self.agents, ResourceKind::Agent, id)
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

    /// Removes every agent, dropping their tags, and returns how many were removed.
    pub fn clear_agents(&self) -> usize {
        self.clear_tagged(&self.agents, ResourceKind::Agent)
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

    /// Removes `id`, returning its card and dropping the workflow's tags.
    #[must_use]
    pub fn remove_workflow(&self, id: &str) -> Option<WorkflowCard> {
        self.remove_tagged(&self.workflows, ResourceKind::Workflow, id)
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

    /// Removes every workflow, dropping their tags, and returns how many were removed.
    pub fn clear_workflows(&self) -> usize {
        self.clear_tagged(&self.workflows, ResourceKind::Workflow)
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

    /// Removes `id`, returning its card and dropping the team's tags.
    #[must_use]
    pub fn remove_team(&self, id: &str) -> Option<TeamCard> {
        self.remove_tagged(&self.teams, ResourceKind::Team, id)
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

    /// Removes every team, dropping their tags, and returns how many were removed.
    pub fn clear_teams(&self) -> usize {
        self.clear_tagged(&self.teams, ResourceKind::Team)
    }

    /// Returns the number of registered teams.
    #[must_use]
    pub fn team_count(&self) -> usize {
        self.teams.len()
    }

    // ------------------------------------------------------------- managers --

    /// Returns the shared tag index (partitioned by [`ResourceKind`]); its
    /// semantics live in [`TagMgr`].
    #[must_use]
    pub fn tags(&self) -> &TagMgr {
        &self.tags
    }

    /// Returns the model factory registry; the semantics live in [`ModelMgr`].
    #[must_use]
    pub fn models(&self) -> &ModelMgr {
        &self.models
    }

    /// Returns the prompt template registry; the semantics live in [`PromptMgr`].
    #[must_use]
    pub fn prompts(&self) -> &PromptMgr {
        &self.prompts
    }

    /// Returns the system-operation registry; see [`SysOperationMgr`].
    #[must_use]
    pub fn sysops(&self) -> &SysOperationMgr {
        &self.sysops
    }

    /// Returns the MCP-server manager; the semantics live in [`McpServerMgr`].
    #[must_use]
    pub fn mcp_servers(&self) -> &McpServerMgr {
        &self.mcp_servers
    }

    // ------------------------------------------------------------- lifecycle --

    /// Returns whether all four run-able registries are empty; the sub-managers
    /// are ignored (see [`ResourceMgr::is_pristine`] for the whole manager).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
            && self.agents.is_empty()
            && self.workflows.is_empty()
            && self.teams.is_empty()
    }

    /// Returns whether the manager is pristine: all four registries **and** all
    /// five sub-managers (including the tag index) are empty.
    #[must_use]
    pub fn is_pristine(&self) -> bool {
        self.is_empty()
            && self.tags.is_empty()
            && self.models.is_empty()
            && self.prompts.is_empty()
            && self.sysops.is_empty()
            && self.mcp_servers.is_empty()
    }

    /// Clears all four registries, resets the five sub-managers (the tag index
    /// included), and zeroes every revision counter.
    ///
    /// Python parity: `Runner::stop` calls `ResourceMgr.release`.
    pub fn release(&self) {
        self.tools.reset();
        self.agents.reset();
        self.workflows.reset();
        self.teams.reset();
        self.tags.reset();
        self.models.reset();
        self.prompts.reset();
        self.sysops.reset();
        self.mcp_servers.reset();
    }

    /// Removes `id` from `registry`, dropping its tags when it was present.
    fn remove_tagged<C: ResourceCard, T>(
        &self,
        registry: &Registry<C, T>,
        kind: ResourceKind,
        id: &str,
    ) -> Option<C> {
        let card = registry.remove(id);
        if card.is_some() {
            let _ = self.tags.remove_resource(kind, id);
        }
        card
    }

    /// Clears `registry`, dropping the tags of every removed entry.
    fn clear_tagged<C: ResourceCard, T>(
        &self,
        registry: &Registry<C, T>,
        kind: ResourceKind,
    ) -> usize {
        let ids = registry.ids();
        let removed = registry.clear();
        if removed > 0 {
            for id in &ids {
                let _ = self.tags.remove_resource(kind, id);
            }
        }
        removed
    }
}

impl std::fmt::Debug for ResourceMgr {
    /// Prints each registry's ids and revision, each sub-manager's count and
    /// revision, and each MCP server's id and state.
    ///
    /// No configuration is rendered, so `auth_headers` and `auth_query_params`
    /// stay private.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResourceMgr")
            .field("tools", &(self.tool_ids(), self.tool_revision()))
            .field("agents", &(self.agent_ids(), self.agent_revision()))
            .field(
                "workflows",
                &(self.workflow_ids(), self.workflow_revision()),
            )
            .field("teams", &(self.team_ids(), self.team_revision()))
            .field("tags", &(self.tags.len(), self.tags.revision()))
            .field("models", &(self.models.len(), self.models.revision()))
            .field("prompts", &(self.prompts.len(), self.prompts.revision()))
            .field("sysops", &(self.sysops.len(), self.sysops.revision()))
            .field(
                "mcp_servers",
                &(self.mcp_servers.list(), self.mcp_servers.revision()),
            )
            .finish_non_exhaustive()
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
    use openjiuwen_foundation::prompt::PromptTemplate;
    use openjiuwen_foundation::tool::mcp::McpServerConfig;
    use openjiuwen_foundation::tool::{FunctionTool, ToolContext, ToolHandler};
    use openjiuwen_multiagent::{handoff_team, HandoffConfig, HandoffTeamConfig};
    use openjiuwen_session::Session;
    use openjiuwen_sysop::{SysOperation, SysOperationCard};
    use openjiuwen_test_support::ScriptedModel;
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

    /// Registers one tool, agent, workflow, and team (`t1`, `a1`, `w1`, `g1`).
    fn one_of_each(manager: &ResourceMgr) {
        manager
            .register_tool(echo_tool(tool_card("t1", "echo")))
            .unwrap();
        let agent = agent_card("a1", "assistant");
        manager
            .register_agent(agent.clone(), agent_provider(agent))
            .unwrap();
        let flow = workflow_card("w1", "flow");
        manager
            .register_workflow(flow.clone(), workflow_provider(flow))
            .unwrap();
        let crew = team_card("g1", "crew");
        manager
            .register_team(crew.clone(), team_provider(crew))
            .unwrap();
    }

    /// Tags every `(kind, id)` pair with the tag `"t"`.
    fn tagged(manager: &ResourceMgr, pairs: &[(ResourceKind, &str)]) {
        for &(kind, id) in pairs {
            manager.tags().tag(kind, id, [Tag::new("t")]).unwrap();
        }
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

    #[test]
    fn removing_a_resource_drops_its_tags() {
        let manager = ResourceMgr::new();
        one_of_each(&manager);
        tagged(&manager, &[(ResourceKind::Tool, "t1")]);
        tagged(&manager, &[(ResourceKind::Agent, "a1")]);
        tagged(&manager, &[(ResourceKind::Workflow, "w1")]);
        tagged(&manager, &[(ResourceKind::Team, "g1")]);

        assert_eq!(manager.remove_tool("t1").unwrap().id, "t1");
        assert_eq!(manager.remove_agent("a1").unwrap().id, "a1");
        assert_eq!(manager.remove_workflow("w1").unwrap().id, "w1");
        assert_eq!(manager.remove_team("g1").unwrap().id, "g1");
        assert!(manager.tags().is_empty(), "tags dropped!");

        // A miss leaves the tag index untouched.
        tagged(&manager, &[(ResourceKind::Tool, "t2")]);
        let revision = manager.tags().revision();
        assert!(manager.remove_tool("t2").is_none());
        assert!(manager.remove_agent("t2").is_none());
        assert!(manager.remove_workflow("t2").is_none());
        assert!(manager.remove_team("t2").is_none());
        assert_eq!(manager.tags().revision(), revision);
        let kept = manager.tags().tags_of(ResourceKind::Tool, "t2");
        assert_eq!(kept, vec![Tag::new("t")]);
    }

    #[test]
    fn clearing_a_kind_drops_the_tags_of_every_removed_resource() {
        let manager = ResourceMgr::new();
        one_of_each(&manager);
        manager
            .register_tool(echo_tool(tool_card("t2", "echo2")))
            .unwrap();
        tagged(&manager, &[(ResourceKind::Tool, "t1")]);
        tagged(&manager, &[(ResourceKind::Tool, "t2")]);
        tagged(&manager, &[(ResourceKind::Agent, "a1")]);

        assert_eq!(manager.clear_tools(), 2);
        assert_eq!(manager.tags().resources(ResourceKind::Tool).len(), 0);
        assert_eq!(manager.tags().resources(ResourceKind::Agent), vec!["a1"]);

        assert_eq!(manager.clear_agents(), 1);
        assert!(manager.tags().list_tags().is_empty(), "collected");
        assert_eq!(manager.clear_workflows(), 1);
        assert_eq!(manager.clear_teams(), 1);
    }

    #[test]
    fn release_resets_every_manager() {
        let manager = ResourceMgr::new();
        one_of_each(&manager);
        manager
            .models()
            .register_model("m1", Arc::new(ScriptedModel::default()))
            .unwrap();
        manager
            .prompts()
            .add_prompt("p1", PromptTemplate::new("hi"))
            .unwrap();
        let operation = SysOperation::new(SysOperationCard::new("op1")).unwrap();
        manager.sysops().register(operation).unwrap();
        manager
            .mcp_servers()
            .register(McpServerConfig::stdio("srv", "cmd"))
            .unwrap();
        tagged(&manager, &[(ResourceKind::Tool, "t1")]);

        assert!(!manager.is_pristine());
        manager.release();

        assert!(manager.is_pristine());
        assert_eq!(manager.tags().revision(), 0);
        assert_eq!(manager.models().revision(), 0);
        assert_eq!(manager.prompts().revision(), 0);
        assert_eq!(manager.sysops().revision(), 0);
        assert_eq!(manager.mcp_servers().revision(), 0);
    }

    #[test]
    fn is_pristine_is_stricter_than_is_empty() {
        let manager = ResourceMgr::new();
        assert!(manager.is_empty() && manager.is_pristine());

        manager
            .models()
            .register_model("m1", Arc::new(ScriptedModel::default()))
            .unwrap();
        assert!(manager.is_empty() && !manager.is_pristine());

        manager
            .prompts()
            .add_prompt("p1", PromptTemplate::new("hi"))
            .unwrap();
        assert!(manager.is_empty() && !manager.is_pristine());
        manager.release();

        tagged(&manager, &[(ResourceKind::Agent, "a1")]);
        assert!(manager.is_empty() && !manager.is_pristine());
    }

    #[test]
    fn debug_output_lists_ids_and_revisions() {
        let manager = ResourceMgr::new();
        one_of_each(&manager);
        manager
            .mcp_servers()
            .register(McpServerConfig::stdio("srv", "cmd").with_server_id("srv-1"))
            .unwrap();
        tagged(&manager, &[(ResourceKind::Tool, "t1")]);

        let rendered = format!("{manager:?}");
        // Each registry prints its ids followed by its revision counter.
        assert!(rendered.contains(r#"(["t1"], 1)"#), "{rendered}");
        assert!(rendered.contains(r#"(["a1"], 1)"#), "{rendered}");
        assert!(rendered.contains(r#"(["w1"], 1)"#), "{rendered}");
        assert!(rendered.contains(r#"(["g1"], 1)"#), "{rendered}");
        // The MCP server is listed by id and state, never by configuration.
        assert!(rendered.contains("srv-1"), "{rendered}");
        assert!(!rendered.contains("auth_headers"), "{rendered}");
    }
}
