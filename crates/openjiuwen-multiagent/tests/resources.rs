#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Public-path tests for the resource-manager seam
//! (`openjiuwen_multiagent::resources`).
//!
//! The suite exercises exactly what a caller outside the crate sees: the
//! [`LocalResourceMgr`] inherent surface and the [`TeamResourceMgr`] trait
//! object. It pins first-registration-wins duplicates, sorted id listings,
//! registration-ordered agent ids, the revision counter, `clear`, and the
//! trait's default methods (a manager that implements only one kind reports
//! every other kind as absent).
//! Everything is in-process: stub tools and models implement the foundation
//! traits directly and members come from the shared `common::FakeAgent`, so the
//! suite is offline and deterministic.

mod common;

use std::collections::BTreeMap;
use std::sync::Arc;

use futures::stream::{self, BoxStream};
use openjiuwen_agent::{Agent, AgentCard};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::{ChatResponse, Model, ModelRequestConfig, StreamChunk};
use openjiuwen_foundation::prompt::BaseMessage;
use openjiuwen_foundation::tool::{Tool, ToolCard, ToolContext};
use openjiuwen_multiagent::resources::{LocalResourceMgr, TeamResourceMgr};
use openjiuwen_multiagent::{AgentProvider, TeamAgent};
use openjiuwen_session::{AgentTeamSession, AgentTeamSessionOptions, Session};
use serde_json::Value;

/// A tool with a caller-chosen card.
struct StubTool {
    /// The card every lookup reports.
    card: ToolCard,
}

impl StubTool {
    /// Builds a tool whose id and name are both `name`.
    fn new(name: &str) -> Arc<Self> {
        Self::with_ids(name, name)
    }

    /// Builds a tool with a distinct card id and name.
    fn with_ids(id: &str, name: &str) -> Arc<Self> {
        Arc::new(Self {
            card: ToolCard::new(name, "a stub tool", serde_json::json!({"type": "object"}))
                .with_id(id),
        })
    }
}

#[async_trait::async_trait]
impl Tool for StubTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, _inputs: Value, _ctx: &ToolContext) -> Result<Value> {
        Ok(serde_json::json!({"ok": true}))
    }
}

/// A model that exists only to be registered; calling it is unsupported.
struct StubModel {
    /// The provider-facing name the registry test asserts on.
    name: &'static str,
}

#[async_trait::async_trait]
impl Model for StubModel {
    // The trait ties the returned provider name to `&self`, so the stub returns
    // a literal through that lifetime rather than a `&'static str`.
    #[allow(clippy::unnecessary_literal_bound)]
    fn provider(&self) -> &str {
        "stub"
    }

    fn model_name(&self) -> &str {
        self.name
    }

    async fn invoke(
        &self,
        _messages: &[BaseMessage],
        _request: &ModelRequestConfig,
    ) -> Result<ChatResponse> {
        Err(Error::unsupported("stub model is registry-only"))
    }

    fn stream(
        &self,
        _messages: Vec<BaseMessage>,
        _request: ModelRequestConfig,
    ) -> BoxStream<'static, Result<StreamChunk>> {
        Box::pin(stream::once(async {
            Err(Error::unsupported("stub model is registry-only"))
        }))
    }
}

/// Builds a model registered under `name`.
fn model(name: &'static str) -> Arc<dyn Model> {
    Arc::new(StubModel { name })
}

/// Builds a provider whose members are called `agent_id`.
fn provider(agent_id: &str) -> AgentProvider {
    let agent: Arc<dyn Agent> = common::FakeAgent::echoing(agent_id);
    Arc::new(move || TeamAgent::new(Arc::clone(&agent)))
}

/// Builds a team session with the deterministic id `session_id`.
fn session(session_id: &str) -> Arc<AgentTeamSession> {
    Arc::new(AgentTeamSession::with_options(AgentTeamSessionOptions {
        session_id: Some(session_id.to_owned()),
        team_id: Some("team".to_owned()),
        ..AgentTeamSessionOptions::default()
    }))
}

/// A manager that knows agent cards and nothing else.
///
/// Every other method keeps the trait default, which is what the
/// `default_trait_methods_*` test asserts.
#[derive(Debug, Default)]
struct CardsOnlyMgr {
    /// The cards this manager publishes.
    cards: BTreeMap<String, AgentCard>,
}

impl TeamResourceMgr for CardsOnlyMgr {
    fn agent_card(&self, agent_id: &str) -> Option<AgentCard> {
        self.cards.get(agent_id).cloned()
    }
}

#[test]
fn tool_registry_round_trip() {
    let mgr = LocalResourceMgr::new();
    assert_eq!(mgr.tool_count(), 0);

    assert!(mgr.register_tool(StubTool::with_ids("t-1", "search")));
    assert_eq!(mgr.tool_count(), 1);
    assert_eq!(mgr.tool("t-1").unwrap().card().name, "search");
    assert_eq!(mgr.tool_by_name("search").unwrap().card().id, "t-1");
    assert_eq!(mgr.tool_card("t-1").unwrap().name, "search");
    assert!(mgr.tool("t-2").is_none());
    assert!(mgr.tool_by_name("missing").is_none());

    let removed = mgr.remove_tool("t-1").unwrap();
    assert_eq!(removed.card().id, "t-1");
    assert!(mgr.remove_tool("t-1").is_none());
    assert_eq!(mgr.tool_count(), 0);
}

#[test]
fn tool_duplicate_keeps_the_first_registration() {
    let mgr = LocalResourceMgr::new();
    assert!(mgr.register_tool(StubTool::with_ids("t-1", "first")));
    assert!(!mgr.register_tool(StubTool::with_ids("t-1", "second")));

    assert_eq!(mgr.tool_count(), 1);
    assert_eq!(mgr.tool_ids(), vec!["t-1".to_owned()]);
    assert_eq!(mgr.tool("t-1").unwrap().card().name, "first");
}

#[test]
fn tool_by_name_prefers_the_lowest_id() {
    let mgr = LocalResourceMgr::new();
    assert!(mgr.register_tool(StubTool::with_ids("b", "same")));
    assert!(mgr.register_tool(StubTool::with_ids("a", "same")));

    assert_eq!(mgr.tool_by_name("same").unwrap().card().id, "a");
    assert_eq!(mgr.tool_ids(), vec!["a".to_owned(), "b".to_owned()]);
}

#[test]
fn tool_ids_are_sorted_and_explicit_ids_are_honoured() {
    let mgr = LocalResourceMgr::new();
    assert!(mgr.register_tool(StubTool::new("zeta")));
    assert!(mgr.register_tool_with("alias", StubTool::with_ids("internal", "alpha")));

    assert_eq!(mgr.tool_ids(), vec!["alias".to_owned(), "zeta".to_owned()]);
    assert_eq!(mgr.tool("alias").unwrap().card().name, "alpha");
    assert!(mgr.tool("internal").is_none());
}

#[test]
fn model_registry_round_trip() {
    let mgr = LocalResourceMgr::new();
    assert!(mgr.register_model("beta", model("beta-model")));
    assert!(mgr.register_model("alpha", model("alpha-model")));
    assert!(!mgr.register_model("alpha", model("shadowed")));

    assert_eq!(mgr.model_count(), 2);
    assert_eq!(mgr.model_ids(), vec!["alpha".to_owned(), "beta".to_owned()]);
    assert_eq!(mgr.model("alpha").unwrap().model_name(), "alpha-model");
    assert_eq!(mgr.model("beta").unwrap().provider(), "stub");
    assert!(mgr.model("gamma").is_none());

    let removed = mgr.remove_model("beta").unwrap();
    assert_eq!(removed.model_name(), "beta-model");
    assert!(mgr.remove_model("beta").is_none());
    assert_eq!(mgr.model_ids(), vec!["alpha".to_owned()]);
}

#[tokio::test]
async fn agent_registry_round_trip() {
    let mgr = LocalResourceMgr::new();
    assert!(mgr.register_agent(AgentCard::new("writer").with_id("w"), provider("w")));
    assert!(mgr.register_agent(AgentCard::new("reviewer").with_id("r"), provider("r")));

    assert_eq!(mgr.agent_count(), 2);
    assert_eq!(mgr.agent_ids(), vec!["w".to_owned(), "r".to_owned()]);
    assert_eq!(mgr.agent_card("w").unwrap().name, "writer");

    let resolved = mgr.agent_provider("r").unwrap()();
    assert_eq!(resolved.agent().card().id, "r");
    let output = resolved
        .agent()
        .invoke(serde_json::json!({"query": "hi"}), None)
        .await
        .unwrap();
    assert_eq!(output.output, "r:hi");

    let removed = mgr.remove_agent("w").unwrap();
    assert_eq!(removed.name, "writer");
    assert!(mgr.remove_agent("w").is_none());
    assert_eq!(mgr.agent_ids(), vec!["r".to_owned()]);
    assert!(mgr.agent_provider("w").is_none());
}

#[test]
fn agent_duplicate_keeps_the_first_card_and_provider() {
    let mgr = LocalResourceMgr::new();
    assert!(mgr.register_agent(AgentCard::new("first").with_id("a"), provider("a")));
    assert!(!mgr.register_agent(AgentCard::new("second").with_id("a"), provider("a2")));

    assert_eq!(mgr.agent_count(), 1);
    assert_eq!(mgr.agent_card("a").unwrap().name, "first");
    assert_eq!(mgr.agent_provider("a").unwrap()().agent().card().id, "a");
}

#[test]
fn session_registry_round_trip() {
    let mgr = LocalResourceMgr::new();
    let first = session("s-1");
    assert_eq!(first.id(), "s-1");
    assert!(mgr.register_team_session(Arc::clone(&first)));
    assert!(mgr.register_team_session(session("s-0")));
    assert!(!mgr.register_team_session(session("s-1")));

    assert_eq!(mgr.session_count(), 2);
    assert_eq!(mgr.session_ids(), vec!["s-0".to_owned(), "s-1".to_owned()]);
    assert!(Arc::ptr_eq(&mgr.team_session("s-1").unwrap(), &first));
    assert!(mgr.team_session("s-2").is_none());

    assert!(mgr.remove_team_session("s-0"));
    assert!(!mgr.remove_team_session("s-0"));
    assert_eq!(mgr.session_ids(), vec!["s-1".to_owned()]);
}

#[test]
fn revision_counts_accepted_mutations_only() {
    let mgr = LocalResourceMgr::new();
    assert_eq!(mgr.revision(), 0);

    assert!(mgr.register_tool(StubTool::new("t")));
    assert!(mgr.register_model("m", model("m")));
    assert!(mgr.register_agent(AgentCard::new("a").with_id("a"), provider("a")));
    assert!(mgr.register_team_session(session("s")));
    assert_eq!(mgr.revision(), 4);

    assert!(!mgr.register_tool(StubTool::new("t")));
    assert!(!mgr.register_model("m", model("m")));
    assert!(!mgr.register_agent(AgentCard::new("a").with_id("a"), provider("a")));
    assert!(!mgr.register_team_session(session("s")));
    assert_eq!(mgr.revision(), 4);

    assert!(mgr.remove_tool("t").is_some());
    assert!(mgr.remove_model("m").is_some());
    assert!(mgr.remove_agent("a").is_some());
    assert!(mgr.remove_team_session("s"));
    assert_eq!(mgr.revision(), 8);

    assert!(mgr.remove_tool("t").is_none());
    assert!(mgr.remove_model("m").is_none());
    assert!(mgr.remove_agent("a").is_none());
    assert!(!mgr.remove_team_session("s"));
    assert_eq!(mgr.revision(), 8);
}

#[test]
fn clear_removes_every_kind_and_bumps_the_revision_once() {
    let mgr = LocalResourceMgr::new();
    assert!(mgr.register_tool(StubTool::new("t")));
    assert!(mgr.register_model("m", model("m")));
    assert!(mgr.register_agent(AgentCard::new("a").with_id("a"), provider("a")));
    assert!(mgr.register_team_session(session("s")));
    let before = mgr.revision();

    assert_eq!(mgr.clear(), 4);
    assert_eq!(mgr.revision(), before + 1);
    assert_eq!(mgr.tool_count(), 0);
    assert_eq!(mgr.model_count(), 0);
    assert_eq!(mgr.agent_count(), 0);
    assert_eq!(mgr.session_count(), 0);
    assert_eq!(mgr.agent_ids(), Vec::<String>::new());
    assert_eq!(mgr.tool_ids(), Vec::<String>::new());
    assert_eq!(mgr.model_ids(), Vec::<String>::new());
    assert_eq!(mgr.session_ids(), Vec::<String>::new());

    assert_eq!(mgr.clear(), 0);
    assert_eq!(mgr.revision(), before + 1);
}

#[test]
fn debug_reports_counts_instead_of_payloads() {
    let mgr = LocalResourceMgr::new();
    assert!(mgr.register_tool(StubTool::new("t")));
    assert!(mgr.register_model("m", model("m")));
    assert!(mgr.register_agent(AgentCard::new("a").with_id("a"), provider("a")));
    assert!(mgr.register_team_session(session("s")));

    let rendered = format!("{mgr:?}");
    assert!(rendered.contains("LocalResourceMgr"));
    assert!(rendered.contains("agents: 1"));
    assert!(rendered.contains("tools: 1"));
    assert!(rendered.contains("models: 1"));
    assert!(rendered.contains("team_sessions: 1"));
    assert!(rendered.contains("revision: 4"));
}

#[test]
fn local_manager_trait_object_delegates_to_the_inherent_methods() {
    let mgr = LocalResourceMgr::new();
    assert!(mgr.register_tool(StubTool::new("t")));
    assert!(mgr.register_model("m", model("m")));
    assert!(mgr.register_agent(AgentCard::new("a").with_id("a"), provider("a")));
    assert!(mgr.register_team_session(session("s")));

    let shared: Arc<dyn TeamResourceMgr> = mgr.clone();
    assert_eq!(shared.agent_card("a").unwrap().id, "a");
    assert!(shared.agent_provider("a").is_some());
    assert!(shared.agent_provider("absent").is_none());
    assert!(shared.tool("t").is_some());
    assert!(shared.tool_by_name("t").is_some());
    assert!(shared.tool("absent").is_none());
    assert!(shared.model("m").is_some());
    assert!(shared.model("absent").is_none());
    assert!(shared.team_session("s").is_some());
    assert!(shared.team_session("absent").is_none());

    // The trait keeps the same first-wins semantics as the inherent methods.
    assert!(!shared.register_agent(AgentCard::new("b").with_id("a"), provider("b")));
    assert_eq!(mgr.agent_card("a").unwrap().name, "a");
    assert!(!shared.register_team_session(session("s")));
    assert!(shared.remove_team_session("s"));
    assert!(mgr.team_session("s").is_none());
}

#[test]
fn default_trait_methods_report_absent() {
    let mut cards = BTreeMap::new();
    cards.insert("known".to_owned(), AgentCard::new("known").with_id("known"));

    let owner = CardsOnlyMgr { cards };
    let dyn_mgr: &dyn TeamResourceMgr = &owner;
    assert_eq!(dyn_mgr.agent_card("known").unwrap().id, "known");
    assert!(dyn_mgr.agent_card("absent").is_none());
    assert!(dyn_mgr.agent_provider("known").is_none());
    assert!(!dyn_mgr.register_agent(AgentCard::new("c").with_id("c"), provider("c")));
    assert!(dyn_mgr.tool("known").is_none());
    assert!(dyn_mgr.tool_by_name("known").is_none());
    assert!(dyn_mgr.model("known").is_none());
    assert!(dyn_mgr.team_session("known").is_none());
    assert!(!dyn_mgr.register_team_session(session("s")));
    assert!(!dyn_mgr.remove_team_session("known"));

    let shared: Arc<dyn TeamResourceMgr> = Arc::new(CardsOnlyMgr::default());
    assert!(shared.agent_card("known").is_none());
    assert!(shared.tool("anything").is_none());
    assert!(format!("{shared:?}").contains("CardsOnlyMgr"));
}
