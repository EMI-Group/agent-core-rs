//! The runner-facing resource seam: a process-level registry of agents, tools,
//! models, and team sessions.
//!
//! Python parity: `TeamRuntime.register_agent` in
//! `openjiuwen/core/multi_agent/team_runtime/team_runtime.py` stores the member
//! card locally *and* calls `Runner.resource_mgr.add_agent(card, provider)`,
//! logging and reusing the existing entry for a duplicate id.
//! `unregister_agent` deliberately leaves the resource manager untouched,
//! because an agent may be shared by several teams.
//!
//! This crate cannot depend on `openjiuwen-runner` (the runner depends on this
//! crate), so the registry is expressed as a trait: [`TeamResourceMgr`] is the
//! lookup surface a team consults before falling back to its own instance-scoped
//! [`TeamRuntime`](crate::TeamRuntime) registry.
//! A runner-backed implementation wraps `openjiuwen_runner::ResourceMgr` and
//! forwards these lookups to its own managers (tools by card id and by name,
//! models by id, agents as cards plus providers, and team sessions), which is how
//! tool, model, and team-session resolution reaches the process-level registry
//! once a runner is configured.
//!
//! [`LocalResourceMgr`] is the standalone implementation: an in-memory,
//! first-registration-wins registry that needs no runner.
//! Everything is synchronous and every mutation takes a short
//! [`parking_lot::RwLock`] write scope, so no lock is ever held across an
//! `.await`; there is no global mutable state and no panic path.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use openjiuwen_agent::AgentCard;
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::tool::{Tool, ToolCard};
use openjiuwen_session::{AgentTeamSession, Session};
use parking_lot::RwLock;

use crate::member::AgentProvider;

/// The runner-facing resource seam: a process-level registry the team consults
/// before falling back to its own instance-scoped registry.
///
/// Every method has a harmless default, so an implementation exposes only the
/// resource kinds it actually owns: an unknown kind or a missing id reports
/// "absent" (or "not registered") instead of failing.
/// Lookups are `&self` and must be cheap; implementations are shared as
/// `Arc<dyn TeamResourceMgr>`, so they are also required to be `Send + Sync`.
///
/// `Debug` is a supertrait for logging, and an implementation must not require
/// `Debug` from the tools and models it stores (print counts or ids instead).
pub trait TeamResourceMgr: Send + Sync + fmt::Debug {
    /// Returns a copy of the card registered under `agent_id`.
    ///
    /// Default: no agents, so `None`.
    fn agent_card(&self, agent_id: &str) -> Option<AgentCard> {
        let _ = agent_id;
        None
    }

    /// Returns the provider registered under `agent_id`.
    ///
    /// Default: no agents, so `None`.
    fn agent_provider(&self, agent_id: &str) -> Option<AgentProvider> {
        let _ = agent_id;
        None
    }

    /// Registers a member; returns `true` when it was added, `false` on a
    /// duplicate.
    ///
    /// Python parity: a duplicate is logged and the existing entry is reused, so
    /// `false` is not an error.
    /// Default: nothing is registered, so `false`.
    fn register_agent(&self, card: AgentCard, provider: AgentProvider) -> bool {
        let _ = (card, provider);
        false
    }

    /// Returns the tool registered under `tool_id` (a [`ToolCard::id`]).
    ///
    /// Default: no tools, so `None`.
    fn tool(&self, tool_id: &str) -> Option<Arc<dyn Tool>> {
        let _ = tool_id;
        None
    }

    /// Returns the tool whose card name is `name`.
    ///
    /// Default: no tools, so `None`.
    fn tool_by_name(&self, name: &str) -> Option<Arc<dyn Tool>> {
        let _ = name;
        None
    }

    /// Returns the model registered under `model_id`.
    ///
    /// Default: no models, so `None`.
    fn model(&self, model_id: &str) -> Option<Arc<dyn Model>> {
        let _ = model_id;
        None
    }

    /// Returns the team session registered under `session_id`.
    ///
    /// Default: no team sessions, so `None`.
    fn team_session(&self, session_id: &str) -> Option<Arc<AgentTeamSession>> {
        let _ = session_id;
        None
    }

    /// Registers a team session keyed on its own id; returns `true` when it was
    /// added, `false` on a duplicate.
    ///
    /// Default: nothing is registered, so `false`.
    fn register_team_session(&self, session: Arc<AgentTeamSession>) -> bool {
        let _ = session;
        false
    }

    /// Removes the team session named `session_id`, reporting whether it existed.
    ///
    /// Default: no team sessions, so `false`.
    fn remove_team_session(&self, session_id: &str) -> bool {
        let _ = session_id;
        false
    }
}

/// Agents in registration order plus their id-keyed entries.
#[derive(Default)]
struct AgentEntries {
    /// Agent ids in registration order (the order [`LocalResourceMgr::agent_ids`]
    /// reports).
    order: Vec<String>,
    /// The card and provider of every registered agent, keyed on the card id.
    entries: HashMap<String, (AgentCard, AgentProvider)>,
}

/// The mutable state of a [`LocalResourceMgr`].
///
/// Each resource kind owns one lock, so a lookup only blocks mutations of the
/// same kind.
#[derive(Default)]
struct ResourceState {
    /// Registered agents, in registration order.
    agents: RwLock<AgentEntries>,
    /// Registered tools keyed on the tool id; `BTreeMap` keeps id listings sorted.
    tools: RwLock<BTreeMap<String, Arc<dyn Tool>>>,
    /// Registered models keyed on the model id.
    models: RwLock<BTreeMap<String, Arc<dyn Model>>>,
    /// Registered team sessions keyed on the session id.
    team_sessions: RwLock<BTreeMap<String, Arc<AgentTeamSession>>>,
    /// Counts every accepted mutation; see [`LocalResourceMgr::revision`].
    revision: AtomicU64,
}

/// The in-memory [`TeamResourceMgr`]: a first-registration-wins registry with no
/// runner and no external service.
///
/// The manager is created as an `Arc` and shared, because every lookup takes
/// `&self` and mutates through the per-kind lock.
/// A duplicate registration returns `false` and leaves the existing entry
/// untouched (Python parity: the duplicate is logged at `debug` level and the
/// existing resource is reused), so callers can treat `false` as "already
/// there".
#[derive(Default)]
pub struct LocalResourceMgr {
    /// The four registries plus the revision counter.
    state: ResourceState,
}

impl fmt::Debug for LocalResourceMgr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalResourceMgr")
            .field("agents", &self.agent_count())
            .field("tools", &self.tool_count())
            .field("models", &self.model_count())
            .field("team_sessions", &self.session_count())
            .field("revision", &self.revision())
            .finish()
    }
}

impl LocalResourceMgr {
    /// Builds an empty manager.
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Registers `tool` under its card id; returns `true` when it was added and
    /// `false` for a duplicate id.
    pub fn register_tool(&self, tool: Arc<dyn Tool>) -> bool {
        let tool_id = tool.card().id.clone();
        self.insert_tool(tool_id, tool)
    }

    /// Registers `tool` under the caller-supplied `tool_id`; returns `true` when
    /// it was added and `false` for a duplicate id.
    ///
    /// The explicit id is for tools whose card id is shared between instances
    /// (for example a per-session clone of one catalog entry).
    pub fn register_tool_with(&self, tool_id: impl Into<String>, tool: Arc<dyn Tool>) -> bool {
        self.insert_tool(tool_id.into(), tool)
    }

    /// Returns the tool registered under `tool_id`.
    #[must_use]
    pub fn tool(&self, tool_id: &str) -> Option<Arc<dyn Tool>> {
        self.state.tools.read().get(tool_id).cloned()
    }

    /// Returns the tool whose card name is `name`.
    ///
    /// Ids are iterated in ascending order, so when two tools share a name the
    /// lexicographically smallest id wins.
    #[must_use]
    pub fn tool_by_name(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.state
            .tools
            .read()
            .values()
            .find(|tool| tool.card().name == name)
            .cloned()
    }

    /// Returns a copy of the card of the tool registered under `tool_id`.
    #[must_use]
    pub fn tool_card(&self, tool_id: &str) -> Option<ToolCard> {
        self.tool(tool_id).map(|tool| tool.card().clone())
    }

    /// Returns the registered tool ids in ascending order.
    #[must_use]
    pub fn tool_ids(&self) -> Vec<String> {
        self.state.tools.read().keys().cloned().collect()
    }

    /// Removes the tool registered under `tool_id`.
    #[must_use]
    pub fn remove_tool(&self, tool_id: &str) -> Option<Arc<dyn Tool>> {
        let removed = self.state.tools.write().remove(tool_id);
        if removed.is_some() {
            self.bump();
        }
        removed
    }

    /// Returns the number of registered tools.
    #[must_use]
    pub fn tool_count(&self) -> usize {
        self.state.tools.read().len()
    }

    /// Registers `model` under `model_id`; returns `true` when it was added and
    /// `false` for a duplicate id.
    pub fn register_model(&self, model_id: impl Into<String>, model: Arc<dyn Model>) -> bool {
        let model_id = model_id.into();
        {
            let mut models = self.state.models.write();
            if models.contains_key(&model_id) {
                drop(models);
                tracing::debug!(
                    model_id = %model_id,
                    "model already registered, reusing the existing entry"
                );
                return false;
            }
            models.insert(model_id, model);
        }
        self.bump();
        true
    }

    /// Returns the model registered under `model_id`.
    #[must_use]
    pub fn model(&self, model_id: &str) -> Option<Arc<dyn Model>> {
        self.state.models.read().get(model_id).cloned()
    }

    /// Returns the registered model ids in ascending order.
    #[must_use]
    pub fn model_ids(&self) -> Vec<String> {
        self.state.models.read().keys().cloned().collect()
    }

    /// Removes the model registered under `model_id`.
    #[must_use]
    pub fn remove_model(&self, model_id: &str) -> Option<Arc<dyn Model>> {
        let removed = self.state.models.write().remove(model_id);
        if removed.is_some() {
            self.bump();
        }
        removed
    }

    /// Returns the number of registered models.
    #[must_use]
    pub fn model_count(&self) -> usize {
        self.state.models.read().len()
    }

    /// Registers `card` with its provider under the card id; returns `true` when
    /// it was added and `false` for a duplicate id.
    ///
    /// The existing card and provider are kept on a duplicate, matching the
    /// Python resource manager's log-and-reuse behaviour.
    pub fn register_agent(&self, card: AgentCard, provider: AgentProvider) -> bool {
        let agent_id = card.id.clone();
        {
            let mut agents = self.state.agents.write();
            if agents.entries.contains_key(&agent_id) {
                drop(agents);
                tracing::debug!(
                    agent_id = %agent_id,
                    "agent already registered, reusing the existing entry"
                );
                return false;
            }
            agents.order.push(agent_id.clone());
            agents.entries.insert(agent_id, (card, provider));
        }
        self.bump();
        true
    }

    /// Returns a copy of the card registered under `agent_id`.
    #[must_use]
    pub fn agent_card(&self, agent_id: &str) -> Option<AgentCard> {
        self.state
            .agents
            .read()
            .entries
            .get(agent_id)
            .map(|(card, _provider)| card.clone())
    }

    /// Returns the provider registered under `agent_id`.
    #[must_use]
    pub fn agent_provider(&self, agent_id: &str) -> Option<AgentProvider> {
        self.state
            .agents
            .read()
            .entries
            .get(agent_id)
            .map(|(_card, provider)| Arc::clone(provider))
    }

    /// Returns the registered agent ids in registration order.
    #[must_use]
    pub fn agent_ids(&self) -> Vec<String> {
        self.state.agents.read().order.clone()
    }

    /// Removes the agent registered under `agent_id`, returning its card.
    ///
    /// An unknown id is a no-op returning `None`.
    #[must_use]
    pub fn remove_agent(&self, agent_id: &str) -> Option<AgentCard> {
        let mut agents = self.state.agents.write();
        let (card, _provider) = agents.entries.remove(agent_id)?;
        agents.order.retain(|id| id != agent_id);
        drop(agents);
        self.bump();
        Some(card)
    }

    /// Returns the number of registered agents.
    #[must_use]
    pub fn agent_count(&self) -> usize {
        self.state.agents.read().order.len()
    }

    /// Registers `session` under its own id; returns `true` when it was added and
    /// `false` for a duplicate id.
    pub fn register_team_session(&self, session: Arc<AgentTeamSession>) -> bool {
        let session_id = session.id().to_owned();
        {
            let mut sessions = self.state.team_sessions.write();
            if sessions.contains_key(&session_id) {
                drop(sessions);
                tracing::debug!(
                    session_id = %session_id,
                    "team session already registered, reusing the existing entry"
                );
                return false;
            }
            sessions.insert(session_id, session);
        }
        self.bump();
        true
    }

    /// Returns the team session registered under `session_id`.
    #[must_use]
    pub fn team_session(&self, session_id: &str) -> Option<Arc<AgentTeamSession>> {
        self.state.team_sessions.read().get(session_id).cloned()
    }

    /// Removes the team session named `session_id`, reporting whether it existed.
    #[must_use]
    pub fn remove_team_session(&self, session_id: &str) -> bool {
        let removed = self.state.team_sessions.write().remove(session_id);
        if removed.is_some() {
            self.bump();
        }
        removed.is_some()
    }

    /// Returns the registered session ids in ascending order.
    #[must_use]
    pub fn session_ids(&self) -> Vec<String> {
        self.state.team_sessions.read().keys().cloned().collect()
    }

    /// Returns the number of registered team sessions.
    #[must_use]
    pub fn session_count(&self) -> usize {
        self.state.team_sessions.read().len()
    }

    /// Returns how many accepted mutations the manager has seen.
    ///
    /// The counter is global rather than per area and starts at `0`, so a host
    /// caching any projection of the registry re-reads it whenever the number
    /// changed.
    /// It advances only when the registry actually changes: a rejected duplicate
    /// or a removal of an unknown id leaves it untouched.
    #[must_use]
    pub fn revision(&self) -> u64 {
        self.state.revision.load(Ordering::Relaxed)
    }

    /// Removes every registered resource, returning how many entries were
    /// removed.
    ///
    /// The revision advances once when at least one entry was removed, so a
    /// `clear` on an empty manager is not a mutation.
    #[must_use]
    pub fn clear(&self) -> usize {
        let agents = std::mem::take(&mut *self.state.agents.write());
        let tools = std::mem::take(&mut *self.state.tools.write());
        let models = std::mem::take(&mut *self.state.models.write());
        let sessions = std::mem::take(&mut *self.state.team_sessions.write());
        let removed = agents.order.len() + tools.len() + models.len() + sessions.len();
        if removed > 0 {
            self.bump();
        }
        removed
    }

    /// Inserts `tool` under `tool_id`, keeping the first registration.
    fn insert_tool(&self, tool_id: String, tool: Arc<dyn Tool>) -> bool {
        {
            let mut tools = self.state.tools.write();
            if tools.contains_key(&tool_id) {
                drop(tools);
                tracing::debug!(
                    tool_id = %tool_id,
                    "tool already registered, reusing the existing entry"
                );
                return false;
            }
            tools.insert(tool_id, tool);
        }
        self.bump();
        true
    }

    /// Advances the revision counter by one.
    fn bump(&self) {
        let _previous = self.state.revision.fetch_add(1, Ordering::Relaxed);
    }
}

impl TeamResourceMgr for LocalResourceMgr {
    fn agent_card(&self, agent_id: &str) -> Option<AgentCard> {
        LocalResourceMgr::agent_card(self, agent_id)
    }

    fn agent_provider(&self, agent_id: &str) -> Option<AgentProvider> {
        LocalResourceMgr::agent_provider(self, agent_id)
    }

    fn register_agent(&self, card: AgentCard, provider: AgentProvider) -> bool {
        LocalResourceMgr::register_agent(self, card, provider)
    }

    fn tool(&self, tool_id: &str) -> Option<Arc<dyn Tool>> {
        LocalResourceMgr::tool(self, tool_id)
    }

    fn tool_by_name(&self, name: &str) -> Option<Arc<dyn Tool>> {
        LocalResourceMgr::tool_by_name(self, name)
    }

    fn model(&self, model_id: &str) -> Option<Arc<dyn Model>> {
        LocalResourceMgr::model(self, model_id)
    }

    fn team_session(&self, session_id: &str) -> Option<Arc<AgentTeamSession>> {
        LocalResourceMgr::team_session(self, session_id)
    }

    fn register_team_session(&self, session: Arc<AgentTeamSession>) -> bool {
        LocalResourceMgr::register_team_session(self, session)
    }

    fn remove_team_session(&self, session_id: &str) -> bool {
        LocalResourceMgr::remove_team_session(self, session_id)
    }
}

#[cfg(test)]
mod tests {
    use openjiuwen_agent::{Agent, ReActAgent};
    use openjiuwen_foundation::tool::ToolContext;
    use openjiuwen_session::AgentTeamSessionOptions;
    use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};
    use serde_json::Value;

    use super::*;
    use crate::member::TeamAgent;

    /// A tool with a fixed card and a trivial success result.
    struct StubTool {
        /// The card every lookup reports.
        card: ToolCard,
    }

    impl StubTool {
        /// Builds a tool named `name` with an id equal to the name.
        fn new(name: &str) -> Arc<Self> {
            let card = ToolCard::new(name, "a stub tool", serde_json::json!({"type": "object"}))
                .with_id(name);
            Arc::new(Self { card })
        }

        /// Builds a tool with a distinct card id and name.
        fn with_ids(id: &str, name: &str) -> Arc<Self> {
            let card = ToolCard::new(name, "a stub tool", serde_json::json!({"type": "object"}))
                .with_id(id);
            Arc::new(Self { card })
        }
    }

    #[async_trait::async_trait]
    impl Tool for StubTool {
        fn card(&self) -> &ToolCard {
            &self.card
        }

        async fn invoke(
            &self,
            _inputs: Value,
            _ctx: &ToolContext,
        ) -> openjiuwen_common::error::Result<Value> {
            Ok(serde_json::json!({"ok": true}))
        }
    }

    /// Builds an agent provider whose members are named `agent_id`.
    fn provider(agent_id: &str) -> AgentProvider {
        let card = AgentCard::new(agent_id).with_id(agent_id);
        let agent: Arc<dyn Agent> = Arc::new(ReActAgent::new(card));
        Arc::new(move || TeamAgent::new(Arc::clone(&agent)))
    }

    /// Builds a model usable as a `dyn Model` placeholder.
    fn model() -> Arc<dyn Model> {
        Arc::new(ScriptedModel::new(std::iter::empty::<ScriptedTurn>()))
    }

    /// Builds a team session with the deterministic id `session_id`.
    fn session(session_id: &str) -> Arc<AgentTeamSession> {
        Arc::new(AgentTeamSession::with_options(AgentTeamSessionOptions {
            session_id: Some(session_id.to_owned()),
            team_id: Some("team".to_owned()),
            ..AgentTeamSessionOptions::default()
        }))
    }

    #[test]
    fn new_manager_is_empty() {
        let mgr = LocalResourceMgr::new();
        assert_eq!(mgr.agent_count(), 0);
        assert_eq!(mgr.tool_count(), 0);
        assert_eq!(mgr.model_count(), 0);
        assert_eq!(mgr.session_count(), 0);
        assert_eq!(mgr.revision(), 0);
        assert!(mgr.agent_card("missing").is_none());
        assert!(mgr.tool("missing").is_none());
        assert!(mgr.model("missing").is_none());
        assert!(mgr.team_session("missing").is_none());
    }

    #[test]
    fn agents_are_first_wins_and_ordered() {
        let mgr = LocalResourceMgr::new();
        assert!(mgr.register_agent(AgentCard::new("a").with_id("a"), provider("a")));
        assert!(mgr.register_agent(AgentCard::new("b").with_id("b"), provider("b")));
        assert!(!mgr.register_agent(AgentCard::new("a2").with_id("a"), provider("a2")));

        assert_eq!(mgr.agent_ids(), vec!["a".to_owned(), "b".to_owned()]);
        assert_eq!(mgr.agent_count(), 2);
        assert_eq!(mgr.agent_card("a").unwrap().name, "a");
        let resolved = mgr.agent_provider("b").unwrap()();
        assert_eq!(resolved.agent().card().id, "b");

        let removed = mgr.remove_agent("a").unwrap();
        assert_eq!(removed.name, "a");
        assert_eq!(mgr.agent_ids(), vec!["b".to_owned()]);
        assert!(mgr.remove_agent("a").is_none());
    }

    #[test]
    fn tools_are_keyed_by_card_id_and_sorted() {
        let mgr = LocalResourceMgr::new();
        assert!(mgr.register_tool(StubTool::new("zeta")));
        assert!(mgr.register_tool(StubTool::new("alpha")));
        assert!(!mgr.register_tool(StubTool::new("alpha")));
        assert!(mgr.register_tool_with("alias", StubTool::with_ids("some-id", "named")));

        assert_eq!(
            mgr.tool_ids(),
            vec!["alias".to_owned(), "alpha".to_owned(), "zeta".to_owned()]
        );
        assert_eq!(mgr.tool_count(), 3);
        assert_eq!(mgr.tool_card("alias").unwrap().name, "named");
        assert!(mgr.tool_by_name("named").is_some());
        assert!(mgr.tool_by_name("absent").is_none());
        assert!(mgr.remove_tool("alias").is_some());
        assert!(mgr.remove_tool("alias").is_none());
        assert_eq!(mgr.tool_count(), 2);
    }

    #[test]
    fn models_are_first_wins_and_sorted() {
        let mgr = LocalResourceMgr::new();
        assert!(mgr.register_model("beta", model()));
        assert!(mgr.register_model("alpha", model()));
        assert!(!mgr.register_model("alpha", model()));
        assert_eq!(mgr.model_ids(), vec!["alpha".to_owned(), "beta".to_owned()]);
        assert_eq!(mgr.model_count(), 2);
        assert!(mgr.model("alpha").is_some());
        assert!(mgr.remove_model("alpha").is_some());
        assert!(mgr.remove_model("alpha").is_none());
    }

    #[test]
    fn team_sessions_are_keyed_by_session_id() {
        let mgr = LocalResourceMgr::new();
        let first = session("s-1");
        assert_eq!(first.id(), "s-1");
        assert!(mgr.register_team_session(Arc::clone(&first)));
        assert!(!mgr.register_team_session(session("s-1")));
        assert!(mgr.register_team_session(session("s-0")));
        assert_eq!(mgr.session_ids(), vec!["s-0".to_owned(), "s-1".to_owned()]);
        assert_eq!(mgr.session_count(), 2);
        assert!(Arc::ptr_eq(&mgr.team_session("s-1").unwrap(), &first));
        assert!(mgr.remove_team_session("s-0"));
        assert!(!mgr.remove_team_session("s-0"));
        assert_eq!(mgr.session_count(), 1);
    }

    #[test]
    fn revision_counts_accepted_mutations_only() {
        let mgr = LocalResourceMgr::new();
        assert_eq!(mgr.revision(), 0);

        assert!(mgr.register_tool(StubTool::new("tool")));
        assert_eq!(mgr.revision(), 1);
        assert!(!mgr.register_tool(StubTool::new("tool")));
        assert_eq!(mgr.revision(), 1);

        assert!(mgr.register_model("m", model()));
        assert!(mgr.register_agent(AgentCard::new("a").with_id("a"), provider("a")));
        assert!(mgr.register_team_session(session("s")));
        assert_eq!(mgr.revision(), 4);

        assert!(mgr.remove_agent("a").is_some());
        assert_eq!(mgr.revision(), 5);
        assert!(mgr.remove_agent("a").is_none());
        assert_eq!(mgr.revision(), 5);
    }

    #[test]
    fn clear_removes_every_kind_and_bumps_once() {
        let mgr = LocalResourceMgr::new();
        assert!(mgr.register_tool(StubTool::new("tool")));
        assert!(mgr.register_model("m", model()));
        assert!(mgr.register_agent(AgentCard::new("a").with_id("a"), provider("a")));
        assert!(mgr.register_team_session(session("s")));
        let before = mgr.revision();

        assert_eq!(mgr.clear(), 4);
        assert_eq!(mgr.revision(), before + 1);
        assert_eq!(mgr.tool_count(), 0);
        assert_eq!(mgr.model_count(), 0);
        assert_eq!(mgr.agent_count(), 0);
        assert_eq!(mgr.session_count(), 0);

        assert_eq!(mgr.clear(), 0);
        assert_eq!(mgr.revision(), before + 1);
    }

    #[test]
    fn debug_reports_counts_without_dumping_payloads() {
        let mgr = LocalResourceMgr::new();
        assert!(mgr.register_tool(StubTool::new("tool")));
        assert!(mgr.register_model("m", model()));
        assert!(mgr.register_agent(AgentCard::new("a").with_id("a"), provider("a")));
        assert!(mgr.register_team_session(session("s")));
        let rendered = format!("{mgr:?}");
        assert!(rendered.contains("LocalResourceMgr"));
        assert!(rendered.contains("tools: 1"));
        assert!(rendered.contains("agents: 1"));
    }

    #[test]
    fn trait_object_lookups_delegate_to_the_local_manager() {
        let mgr = LocalResourceMgr::new();
        assert!(mgr.register_tool(StubTool::new("tool")));
        assert!(mgr.register_model("m", model()));
        assert!(mgr.register_agent(AgentCard::new("a").with_id("a"), provider("a")));
        assert!(mgr.register_team_session(session("s")));

        let dyn_mgr: &dyn TeamResourceMgr = &*mgr;
        assert_eq!(dyn_mgr.agent_card("a").unwrap().id, "a");
        assert!(dyn_mgr.agent_provider("a").is_some());
        assert!(dyn_mgr.agent_provider("missing").is_none());
        assert!(dyn_mgr.tool("tool").is_some());
        assert!(dyn_mgr.tool_by_name("tool").is_some());
        assert!(dyn_mgr.model("m").is_some());
        assert!(dyn_mgr.team_session("s").is_some());
        assert!(!dyn_mgr.register_team_session(session("s")));
        assert!(dyn_mgr.remove_team_session("s"));

        let shared: Arc<dyn TeamResourceMgr> = mgr.clone();
        assert!(shared.tool("tool").is_some());
    }
}
