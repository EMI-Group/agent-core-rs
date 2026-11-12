#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration suite for [`ResourceMgr`]: the four typed registries, duplicate
//! policies, provider-per-get semantics, name lookups, removal, revisions,
//! `release`, concurrent registration, and card JSON round-trips.
//!
//! Every test builds its own `ResourceMgr::new()`; nothing here touches the
//! network, the filesystem, or the clock.
//!
//! Ports `tests/unit_tests/core/runner/test_resource_manager.py`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use futures::future::BoxFuture;
use futures::stream::BoxStream;
use openjiuwen_agent::{Agent, AgentCard, AgentChunk, AgentOutput};
use openjiuwen_common::error::{ErrorKind, Result};
use openjiuwen_foundation::tool::{FunctionTool, Tool, ToolCard, ToolContext, ToolHandler};
use openjiuwen_multiagent::{handoff_team, HandoffConfig, HandoffTeamConfig, Team, TeamCard};
use openjiuwen_runner::resources::{
    workflow_key, RegisterPolicy, Registration, ResourceCard, ResourceMgr,
};
use openjiuwen_session::Session;
use openjiuwen_workflow::{Workflow, WorkflowCard};
use serde_json::{json, Value};

// ----------------------------------------------------------------- fixtures --

/// An agent that answers with its card id.
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

/// Builds an agent card with an explicit id and name.
fn agent_card(id: &str, name: &str) -> AgentCard {
    AgentCard::default().with_id(id).with_name(name)
}

/// An agent provider minting a fresh [`FakeAgent`] per lookup.
fn agent_provider(card: AgentCard) -> Arc<dyn Fn() -> Arc<dyn Agent> + Send + Sync> {
    Arc::new(move || Arc::new(FakeAgent { card: card.clone() }) as Arc<dyn Agent>)
}

/// A provider that counts its invocations.
fn counting_agent_provider(
    card: AgentCard,
    calls: &Arc<AtomicUsize>,
) -> Arc<dyn Fn() -> Arc<dyn Agent> + Send + Sync> {
    let calls = Arc::clone(calls);
    Arc::new(move || {
        calls.fetch_add(1, Ordering::SeqCst);
        Arc::new(FakeAgent { card: card.clone() }) as Arc<dyn Agent>
    })
}

/// A workflow provider minting a fresh (empty) workflow per lookup.
fn workflow_provider(card: WorkflowCard) -> Arc<dyn Fn() -> Workflow + Send + Sync> {
    Arc::new(move || Workflow::new(card.clone()))
}

/// A team provider minting a fresh handoff team per lookup.
fn team_provider(card: TeamCard) -> Arc<dyn Fn() -> Arc<dyn Team> + Send + Sync> {
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

/// A tool card with an explicit id and name.
fn tool_card(id: &str, name: &str) -> ToolCard {
    ToolCard::new(name, "echoes its inputs", json!({})).with_id(id)
}

/// A workflow card with an explicit id and name.
fn workflow_card(id: &str, name: &str) -> WorkflowCard {
    WorkflowCard {
        id: id.to_owned(),
        name: name.to_owned(),
        ..WorkflowCard::default()
    }
}

/// A team card with an explicit id and name.
fn team_card(id: &str, name: &str) -> TeamCard {
    TeamCard::new(name).with_id(id)
}

/// Registers one resource of every kind.
fn populated() -> Arc<ResourceMgr> {
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
    manager
}

// -------------------------------------------------------------------- tests --

// ports tests/unit_tests/core/runner/test_resource_manager.py
#[test]
fn tools_register_look_up_and_remove() {
    let manager = ResourceMgr::new();
    let tool = echo_tool(tool_card("t1", "echo"));
    manager.register_tool(Arc::clone(&tool)).unwrap();
    manager
        .register_tool(echo_tool(tool_card("t2", "other")))
        .unwrap();

    assert_eq!(manager.tool_count(), 2);
    assert_eq!(manager.tool_ids(), vec!["t1", "t2"]);
    assert_eq!(manager.tool_card("t1").unwrap().name, "echo");
    assert_eq!(
        manager.tool_card("t1").unwrap().description,
        "echoes its inputs"
    );
    assert_eq!(manager.tool_by_name("other").unwrap().card().id, "t2");

    // Tools are registered from a live instance: the same `Arc` comes back.
    let looked_up = manager.tool("t1").unwrap();
    assert!(Arc::ptr_eq(&looked_up, &tool));

    let removed = manager.remove_tool("t1").unwrap();
    assert_eq!(removed.id, "t1");
    assert!(manager.tool("t1").is_none());
    assert_eq!(manager.tool_count(), 1);
    assert_eq!(manager.tool_revision(), 3);
}

// ports tests/unit_tests/core/runner/test_resource_manager.py
#[test]
fn agents_invoke_the_provider_on_every_lookup() {
    let manager = ResourceMgr::new();
    let calls = Arc::new(AtomicUsize::new(0));
    manager
        .register_agent(
            agent_card("a1", "assistant"),
            counting_agent_provider(agent_card("a1", "assistant"), &calls),
        )
        .unwrap();

    assert!(manager.agent("missing").is_none());
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "a missing id resolves nothing"
    );

    let first = manager.agent("a1").unwrap();
    let second = manager.agent("a1").unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(
        !Arc::ptr_eq(&first, &second),
        "each lookup mints a fresh agent"
    );
    assert_eq!(first.card().id, "a1");
}

// ports tests/unit_tests/core/runner/test_resource_manager.py
#[test]
fn workflows_and_teams_register_and_look_up() {
    let manager = ResourceMgr::new();
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

    assert_eq!(manager.workflow_count(), 1);
    assert_eq!(manager.workflow("w1").unwrap().card().name, "flow");
    assert_eq!(manager.workflow_by_name("flow").unwrap().card().id, "w1");
    assert_eq!(manager.workflow_card("w1").unwrap().name, "flow");

    assert_eq!(manager.team_count(), 1);
    assert_eq!(manager.team("g1").unwrap().card().name, "crew");
    assert_eq!(manager.team_by_name("crew").unwrap().card().id, "g1");

    let removed = manager.remove_workflow("w1").unwrap();
    assert_eq!(removed.id, "w1");
    let removed = manager.remove_team("g1").unwrap();
    assert_eq!(removed.id, "g1");
    assert_eq!(manager.workflow_revision(), 2);
    assert_eq!(manager.team_revision(), 2);
}

// ports tests/unit_tests/core/runner/test_resource_manager.py
#[test]
fn fail_policy_keeps_the_first_and_reports_validation() {
    let manager = ResourceMgr::new();
    manager
        .register_agent(
            agent_card("a1", "assistant"),
            agent_provider(agent_card("a1", "assistant")),
        )
        .unwrap();

    let error = manager
        .register_agent(
            agent_card("a1", "duplicate"),
            agent_provider(agent_card("a1", "duplicate")),
        )
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(
        error.to_string(),
        "validation: agent 'a1' is already registered"
    );

    assert_eq!(manager.agent_count(), 1);
    assert_eq!(manager.agent_card("a1").unwrap().name, "assistant");
    assert_eq!(
        manager.agent_revision(),
        1,
        "the rejected write did not mutate"
    );
}

// ports tests/unit_tests/core/runner/test_resource_manager.py
#[test]
fn keep_policy_leaves_the_original() {
    let manager = ResourceMgr::new();
    manager
        .register_agent(
            agent_card("a1", "assistant"),
            agent_provider(agent_card("a1", "assistant")),
        )
        .unwrap();

    let registration = manager
        .register_agent_with(
            agent_card("a1", "ignored"),
            agent_provider(agent_card("a1", "ignored")),
            RegisterPolicy::Keep,
        )
        .unwrap();
    assert_eq!(
        registration,
        Registration {
            id: "a1".to_owned(),
            name: "assistant".to_owned(),
            replaced: false,
        }
    );
    assert_eq!(manager.agent_card("a1").unwrap().name, "assistant");
    assert_eq!(manager.agent_revision(), 1);
}

// ports tests/unit_tests/core/runner/test_resource_manager.py
#[test]
fn replace_policy_swaps_the_card_and_the_provider() {
    let manager = ResourceMgr::new();
    manager
        .register_agent(
            agent_card("a1", "assistant"),
            agent_provider(agent_card("a1", "assistant")),
        )
        .unwrap();

    let registration = manager
        .register_agent_with(
            agent_card("a1", "replacement"),
            agent_provider(agent_card("a1", "replacement")),
            RegisterPolicy::Replace,
        )
        .unwrap();
    assert_eq!(
        registration,
        Registration {
            id: "a1".to_owned(),
            name: "replacement".to_owned(),
            replaced: true,
        }
    );
    assert_eq!(manager.agent_card("a1").unwrap().name, "replacement");
    assert_eq!(
        manager.agent_count(),
        1,
        "replace does not grow the registry"
    );
    assert_eq!(manager.agent_revision(), 2);

    // The tool facade exposes the same policy.
    manager
        .register_tool_with(echo_tool(tool_card("t1", "echo")), RegisterPolicy::Fail)
        .unwrap();
    let replaced = manager
        .register_tool_with(echo_tool(tool_card("t1", "echo2")), RegisterPolicy::Replace)
        .unwrap();
    assert!(replaced.replaced);
    assert_eq!(manager.tool_card("t1").unwrap().name, "echo2");
}

// ports tests/unit_tests/core/runner/test_resource_manager.py
#[test]
fn card_and_name_lookups_prefer_the_first_registration() {
    let manager = ResourceMgr::new();
    for id in ["a1", "a2", "a3"] {
        manager
            .register_agent(
                agent_card(id, "shared"),
                agent_provider(agent_card(id, "shared")),
            )
            .unwrap();
    }

    assert_eq!(manager.agent_card("a2").unwrap().id, "a2");
    assert_eq!(
        manager.agent_by_name("shared").unwrap().card().id,
        "a1",
        "names are not unique; the first registration wins"
    );
    assert!(manager.agent_by_name("absent").is_none());

    let cards = manager.agents();
    let ids: Vec<&str> = cards.iter().map(|card| card.id.as_str()).collect();
    assert_eq!(
        ids,
        vec!["a1", "a2", "a3"],
        "registration order is preserved"
    );
}

// ports tests/unit_tests/core/runner/test_resource_manager.py
#[test]
fn removal_returns_the_card_and_updates_the_revision() {
    let manager = ResourceMgr::new();
    for id in ["a1", "a2", "a3"] {
        manager
            .register_agent(agent_card(id, id), agent_provider(agent_card(id, id)))
            .unwrap();
    }
    assert_eq!(manager.agent_revision(), 3);

    let removed = manager.remove_agent("a2").unwrap();
    assert_eq!(removed.id, "a2");
    assert!(manager.agent("a2").is_none());
    assert_eq!(manager.agent_ids(), vec!["a1", "a3"]);
    assert_eq!(manager.agent_revision(), 4);
    assert!(
        manager.agent("a3").is_some(),
        "the shifted position still resolves"
    );

    assert!(manager.remove_agent("a2").is_none());
    assert_eq!(manager.agent_revision(), 4);
}

// ports tests/unit_tests/core/runner/test_resource_manager.py
#[test]
fn clear_per_kind_reports_counts_and_bumps_only_that_revision() {
    let manager = populated();
    assert_eq!(manager.tool_revision(), 1);
    assert_eq!(manager.agent_revision(), 1);

    assert_eq!(manager.clear_agents(), 1);
    assert_eq!(manager.agent_count(), 0);
    assert_eq!(manager.agent_revision(), 2);
    assert_eq!(manager.tool_revision(), 1, "other kinds are untouched");
    assert_eq!(manager.clear_agents(), 0);
    assert_eq!(manager.agent_revision(), 2);

    assert_eq!(manager.clear_tools(), 1);
    assert_eq!(manager.clear_workflows(), 1);
    assert_eq!(manager.clear_teams(), 1);
    assert!(manager.is_empty());
}

// ports tests/unit_tests/core/runner/test_resource_manager.py
#[test]
fn release_clears_every_registry_and_resets_revisions() {
    let manager = populated();
    assert!(!manager.is_empty());

    manager.release();

    assert!(manager.is_empty());
    assert_eq!(manager.tool_count(), 0);
    assert_eq!(manager.agent_count(), 0);
    assert_eq!(manager.workflow_count(), 0);
    assert_eq!(manager.team_count(), 0);
    assert_eq!(manager.tool_revision(), 0);
    assert_eq!(manager.agent_revision(), 0);
    assert_eq!(manager.workflow_revision(), 0);
    assert_eq!(manager.team_revision(), 0);
}

// ports tests/unit_tests/core/runner/test_resource_manager.py
#[test]
fn empty_and_whitespace_ids_are_rejected() {
    let manager = ResourceMgr::new();

    let error = manager
        .register_agent(
            agent_card("   ", "ghost"),
            agent_provider(agent_card("a1", "ghost")),
        )
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(
        error.to_string(),
        "validation: agent id cannot be empty or whitespace only"
    );

    let error = manager
        .register_tool(echo_tool(tool_card("", "ghost")))
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "validation: tool id cannot be empty or whitespace only"
    );

    assert!(manager.is_empty());
    assert_eq!(manager.agent_revision(), 0);
}

#[test]
fn the_same_id_is_independent_across_kinds() {
    let manager = ResourceMgr::new();
    manager
        .register_agent(
            agent_card("shared", "agent"),
            agent_provider(agent_card("shared", "agent")),
        )
        .unwrap();
    manager
        .register_tool(echo_tool(tool_card("shared", "tool")))
        .unwrap();

    assert_eq!(manager.agent_card("shared").unwrap().name, "agent");
    assert_eq!(manager.tool_card("shared").unwrap().name, "tool");
    assert_eq!(manager.agent_count(), 1);
    assert_eq!(manager.tool_count(), 1);
}

#[test]
fn every_card_implements_resource_card() {
    let tool = tool_card("t1", "echo");
    let agent = agent_card("a1", "assistant");
    let workflow = workflow_card("w1", "flow");
    let team = team_card("g1", "crew");

    assert_eq!((ResourceCard::id(&tool), tool.name()), ("t1", "echo"));
    assert_eq!(
        (ResourceCard::id(&agent), agent.name()),
        ("a1", "assistant")
    );
    assert_eq!(
        (ResourceCard::id(&workflow), workflow.name()),
        ("w1", "flow")
    );
    assert_eq!((ResourceCard::id(&team), team.name()), ("g1", "crew"));
}

#[test]
fn workflow_key_composes_id_and_version() {
    assert_eq!(workflow_key("wf", "1.0.0"), "wf_1.0.0");
    assert_eq!(workflow_key("wf", ""), "wf_");
}

#[test]
fn cards_round_trip_through_json() {
    let cards = [
        serde_json::to_value(tool_card("t1", "echo")).unwrap(),
        serde_json::to_value(agent_card("a1", "assistant")).unwrap(),
        serde_json::to_value(workflow_card("w1", "flow")).unwrap(),
        serde_json::to_value(team_card("g1", "crew")).unwrap(),
    ];

    let tool: ToolCard = serde_json::from_value(cards[0].clone()).unwrap();
    assert_eq!(tool, tool_card("t1", "echo"));
    let agent: AgentCard = serde_json::from_value(cards[1].clone()).unwrap();
    assert_eq!(agent, agent_card("a1", "assistant"));
    let workflow: WorkflowCard = serde_json::from_value(cards[2].clone()).unwrap();
    assert_eq!(workflow, workflow_card("w1", "flow"));
    let team: TeamCard = serde_json::from_value(cards[3].clone()).unwrap();
    assert_eq!(team, team_card("g1", "crew"));

    // An unknown field next to a known card is tolerated.
    let mut value = cards[1].clone();
    value["future_field"] = json!(7);
    let agent: AgentCard = serde_json::from_value(value).unwrap();
    assert_eq!(agent.id, "a1");
}

#[test]
fn registration_round_trips_through_json() {
    let registration = Registration {
        id: "a1".to_owned(),
        name: "assistant".to_owned(),
        replaced: true,
    };
    let text = serde_json::to_string(&registration).unwrap();
    assert_eq!(text, r#"{"id":"a1","name":"assistant","replaced":true}"#);
    let back: Registration = serde_json::from_str(&text).unwrap();
    assert_eq!(back, registration);
}

// ports tests/unit_tests/core/runner/test_resource_manager.py
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_registration_is_serialized() {
    let manager = ResourceMgr::new();
    let mut handles = Vec::new();

    for worker in 0..8 {
        let manager = Arc::clone(&manager);
        handles.push(tokio::spawn(async move {
            for index in 0..16 {
                let id = format!("a{worker}-{index}");
                manager
                    .register_agent(agent_card(&id, &id), agent_provider(agent_card(&id, &id)))
                    .unwrap();
            }
        }));
    }
    for handle in handles {
        handle.await.unwrap();
    }

    assert_eq!(manager.agent_count(), 128);
    assert_eq!(manager.agent_revision(), 128);
    let mut ids = manager.agent_ids();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), 128, "no registration was lost or overwritten");
}

// ports tests/unit_tests/core/runner/test_resource_manager.py
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_duplicate_fail_policy_keeps_exactly_one() {
    let manager = ResourceMgr::new();
    let mut handles = Vec::new();

    for _ in 0..8 {
        let manager = Arc::clone(&manager);
        handles.push(tokio::spawn(async move {
            manager.register_agent(
                agent_card("only", "only"),
                agent_provider(agent_card("only", "only")),
            )
        }));
    }

    let mut accepted = 0;
    for handle in handles {
        if handle.await.unwrap().is_ok() {
            accepted += 1;
        }
    }
    assert_eq!(accepted, 1);
    assert_eq!(manager.agent_count(), 1);
    assert_eq!(manager.agent_revision(), 1);
}
