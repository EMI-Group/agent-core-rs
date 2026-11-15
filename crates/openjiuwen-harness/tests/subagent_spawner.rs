//! Integration suite for the production [`RuntimeSubAgentSpawner`].
//!
//! Covers the two halves of the wiring this adapter closes:
//!
//! - a `DeepAgent` built with sub-agents exposes a working [`SubAgentSpawner`] over its own
//!   runtime (the real `DeepAgentSubAgentFactory` builds the child, which answers with a
//!   `ScriptedModel`), and a caller-supplied `subagent` rail still wins;
//! - the adapter's own concurrency ledger: a saturated spawner rejects the extra spawn with a
//!   typed limit error and reuses the slot once the instance is final.
//!
//! Everything is offline and deterministic: a temp workspace, a scripted model, and one
//! gated fake agent whose only wait is released explicitly. Awaits are bounded at 10 s.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::time::Duration;

use futures::stream::BoxStream;
use openjiuwen_agent::{Agent, AgentCard, AgentChunk, AgentOutput};
use openjiuwen_common::error::Result;
use openjiuwen_foundation::llm::Model;
use openjiuwen_harness::agent::DeepAgentBuilder;
use openjiuwen_harness::rails::{SpawnRequest, SubAgentSpawner, SubAgentState, SubagentRail};
use openjiuwen_harness::schema::{DeepAgentConfig, SubAgentConfig};
use openjiuwen_harness::subagents::{
    general_purpose_config, RuntimeSubAgentSpawner, SubAgentFactory, SubAgentRegistry,
    SubAgentRuntime, SubAgentSpec,
};
use openjiuwen_harness::workspace::Workspace;
use openjiuwen_session::Session;
use openjiuwen_test_support::{ScriptedModel, TempWorkspace};
use serde_json::Value;
use tokio::sync::Notify;

/// Bounded deadline for a wait a scripted child satisfies on its first turn.
const BOUND: Duration = Duration::from_secs(10);

// ---------------------------------------------------------------------------
// Fakes for the concurrency-ledger suite.
// ---------------------------------------------------------------------------

/// An [`Agent`] whose only turn blocks until the gate is notified.
struct GatedAgent {
    /// Card copied from the configuration the factory received.
    card: AgentCard,
    /// Released by the test to let the turn finish.
    gate: Arc<Notify>,
}

#[async_trait::async_trait]
impl Agent for GatedAgent {
    fn card(&self) -> &AgentCard {
        &self.card
    }

    async fn invoke(
        &self,
        _inputs: Value,
        _session: Option<Arc<dyn Session>>,
    ) -> Result<AgentOutput> {
        self.gate.notified().await;
        Ok(AgentOutput::answer("gated answer"))
    }

    fn stream(
        &self,
        _inputs: Value,
        _session: Option<Arc<dyn Session>>,
    ) -> BoxStream<'static, Result<AgentChunk>> {
        Box::pin(futures::stream::empty())
    }
}

/// A [`SubAgentFactory`] that builds [`GatedAgent`]s sharing one gate.
struct GatedFactory {
    /// Gate every child turn waits on.
    gate: Arc<Notify>,
}

#[async_trait::async_trait]
impl SubAgentFactory for GatedFactory {
    async fn create(&self, config: &SubAgentConfig, _session_id: &str) -> Result<Arc<dyn Agent>> {
        Ok(Arc::new(GatedAgent {
            card: config.agent_card.clone(),
            gate: Arc::clone(&self.gate),
        }))
    }
}

/// Builds an English registry with the built-in presets plus the general-purpose fallback.
fn gated_runtime(gate: Arc<Notify>) -> Arc<SubAgentRuntime> {
    let registry = SubAgentRegistry::with_builtin_presets("en");
    registry.register(SubAgentSpec::new(general_purpose_config()));
    Arc::new(SubAgentRuntime::new(
        Arc::new(registry),
        Arc::new(GatedFactory { gate }),
        "parent-session",
    ))
}

// ---------------------------------------------------------------------------
// Builder wiring.
// ---------------------------------------------------------------------------

/// The builder wires the production spawner into the default subagent rail.
#[tokio::test]
async fn a_built_agent_exposes_a_spawner_over_its_subagent_runtime() {
    let temp = TempWorkspace::new().unwrap();
    let workspace = Workspace::for_language(temp.workspace(), "en");
    let model: Arc<dyn Model> = Arc::new(ScriptedModel::text("child answer"));
    let config = DeepAgentConfig::default()
        .with_language("en".to_string())
        .with_workspace(workspace)
        .with_subagents(vec![SubAgentConfig::new(AgentCard::new("helper"))])
        .with_add_general_purpose_agent(true);
    let agent = DeepAgentBuilder::new(AgentCard::new("parent"))
        .config(config)
        .model(model)
        .build()
        .unwrap();

    assert!(agent.rails().names().iter().any(|name| name == "subagent"));
    let spawner = agent
        .subagent_spawner()
        .expect("the default subagent rail carries the production spawner");

    let handle = spawner
        .spawn(SpawnRequest::new("do the task").with_preset("helper"))
        .await
        .unwrap();
    assert_eq!(handle.preset.as_deref(), Some("helper"));
    assert!(!handle.id.is_empty());

    let status = spawner.wait(&handle.id, Some(BOUND)).await.unwrap();
    assert_eq!(status.state, SubAgentState::Completed);
    assert!(status
        .output
        .as_deref()
        .unwrap_or_default()
        .contains("child answer"));

    // `list` sees the live instance with its runtime status.
    let listed = spawner.list().await.unwrap();
    assert!(listed.iter().any(|entry| entry.id == handle.id
        && entry.state == SubAgentState::Completed
        && entry.output.is_none()));

    // Closing a finished instance succeeds and the tombstone still answers.
    spawner.close(&handle.id).await.unwrap();
    let closed = spawner.wait(&handle.id, Some(BOUND)).await.unwrap();
    assert_eq!(closed.state, SubAgentState::Closed);
    assert!(spawner.list().await.unwrap().is_empty());
}

/// A caller rail named `subagent` replaces the default one and keeps its own seam.
#[tokio::test]
async fn a_caller_subagent_rail_wins() {
    let temp = TempWorkspace::new().unwrap();
    let workspace = Workspace::for_language(temp.workspace(), "en");
    let model: Arc<dyn Model> = Arc::new(ScriptedModel::text("unused"));
    let config = DeepAgentConfig::default()
        .with_language("en".to_string())
        .with_workspace(workspace)
        .with_add_general_purpose_agent(true);
    let agent = DeepAgentBuilder::new(AgentCard::new("parent"))
        .config(config)
        .model(model)
        .rail(Arc::new(SubagentRail::new()))
        .build()
        .unwrap();

    assert!(agent.rails().names().iter().any(|name| name == "subagent"));
    assert!(agent.subagent_spawner().is_none());
}

/// Without mounted sub-agents there is no runtime, so there is no spawner.
#[tokio::test]
async fn an_agent_without_subagents_has_no_spawner() {
    let temp = TempWorkspace::new().unwrap();
    let workspace = Workspace::for_language(temp.workspace(), "en");
    let model: Arc<dyn Model> = Arc::new(ScriptedModel::text("unused"));
    let agent = DeepAgentBuilder::new(AgentCard::new("parent"))
        .config(
            DeepAgentConfig::default()
                .with_language("en".to_string())
                .with_workspace(workspace),
        )
        .model(model)
        .build()
        .unwrap();
    assert!(agent.subagent_spawner().is_none());
}

// ---------------------------------------------------------------------------
// Concurrency ledger.
// ---------------------------------------------------------------------------

/// One live spawn holds the only permit; a second one fails fast.
#[tokio::test]
async fn the_concurrency_limit_rejects_a_second_live_spawn() {
    let gate = Arc::new(Notify::new());
    let runtime = gated_runtime(Arc::clone(&gate));
    let rail =
        SubagentRail::new().with_spawner(Arc::new(RuntimeSubAgentSpawner::with_limit(runtime, 1)));
    let seam: Arc<dyn SubAgentSpawner> = rail.spawner().unwrap().clone();

    // The gated turn keeps the first instance live, so the permit stays taken.
    let first = seam.spawn(SpawnRequest::new("first")).await.unwrap();
    let error = seam
        .spawn(SpawnRequest::new("second"))
        .await
        .expect_err("the second spawn must be rejected");
    assert!(error
        .to_string()
        .contains("subagent concurrency limit reached"));

    gate.notify_one();
    let status = seam.wait(&first.id, Some(BOUND)).await.unwrap();
    assert_eq!(status.state, SubAgentState::Completed);
    assert_eq!(status.output.as_deref(), Some("gated answer"));

    // The slot is reusable once the first instance is final.
    let second = seam.spawn(SpawnRequest::new("again")).await.unwrap();
    gate.notify_one();
    let status = seam.wait(&second.id, Some(BOUND)).await.unwrap();
    assert_eq!(status.state, SubAgentState::Completed);
}

/// An unknown id is a typed `NotFound`, not a silent closed status.
#[tokio::test]
async fn an_unknown_id_is_reported_as_not_found() {
    let gate = Arc::new(Notify::new());
    let spawner = RuntimeSubAgentSpawner::new(gated_runtime(gate));
    let error = spawner.wait("ghost", Some(BOUND)).await.unwrap_err();
    assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::NotFound);
    let error = spawner.close("ghost").await.unwrap_err();
    assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::NotFound);
}
