//! Offline coverage for the goal rail's event path, driven through the crate root.
//!
//! The goal rail queues `goal.updated` under [`GOAL_EVENT_EXTRA_KEY`] and the `DeepAgent`'s
//! `RailEventDrain` forwards whatever is queued to the invocation stream and to
//! [`openjiuwen_harness::DeepAgent::drain_rail_events`]. These suites use the real
//! [`GoalStore`] and [`GetCurrentGoalTool`] in a temp workspace with a `ScriptedModel`, so the
//! whole path — auto registration, task iteration, drain, stream — runs without a network.
//!
//! The rail is mounted the way the builder does it: a goal tool card is on the agent, so
//! `DeepAgentBuilder::build` registers `GoalRail` automatically. That exercises the production
//! wiring rather than a hand-built rail.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use anyhow::{ensure, Result};
use chrono::{TimeZone, Utc};
use futures::StreamExt;
use openjiuwen_agent::output::INTERACTION;
use openjiuwen_agent::{Agent, AgentCard, AgentChunk};
use openjiuwen_common::error::Result as CommonResult;
use openjiuwen_harness::{
    DeepAgentBuilder, DeepAgentConfig, GetCurrentGoalTool, GoalAssessment, GoalAssessmentStatus,
    GoalRail, GoalRailConfig, GoalStore, HarnessRail, HarnessRailContext, InteractionEvent,
    TaskIterationInfo, Workspace, GOAL_EVENT_EXTRA_KEY,
};
use openjiuwen_test_support::ScriptedModel;
use serde_json::{json, Value};

/// A fixed instant, so the persisted goal timestamps never depend on the wall clock.
fn now() -> chrono::DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 1, 2, 3, 4, 5)
        .single()
        .expect("the fixture instant is unambiguous")
}

/// Creates a store whose goal has already settled.
async fn settled_store(workspace: &Workspace) -> Result<Arc<GoalStore>> {
    let store = GoalStore::for_workspace(workspace);
    store.start("ship the goal rail", now()).await?;
    let assessment = GoalAssessment::new(GoalAssessmentStatus::Complete, "tests pass", None, None);
    let record = store
        .update(|record| record.record_progress(&assessment, now()), now())
        .await?
        .expect("a persisted goal");
    ensure!(
        record.status.is_terminal(),
        "the fixture goal must be terminal, got {:?}",
        record.status
    );
    Ok(store)
}

/// Builds a task-loop `DeepAgent` over `workspace` with the `get_current_goal` tool mounted.
fn goal_agent(
    workspace: Workspace,
    store: Arc<GoalStore>,
) -> Result<Arc<openjiuwen_harness::DeepAgent>> {
    Ok(DeepAgentBuilder::new(AgentCard::new("goal-agent"))
        .config(
            DeepAgentConfig::default()
                .with_language("en".to_string())
                .with_workspace(workspace)
                .with_enable_task_loop(true),
        )
        .model(Arc::new(ScriptedModel::text("done")))
        .tool(Arc::new(GetCurrentGoalTool::new(store, "en")))
        .build()?)
}

/// Returns the `__interaction__` payloads of `chunks`, in order.
fn interaction_frames(chunks: &[CommonResult<AgentChunk>]) -> Vec<Value> {
    chunks
        .iter()
        .filter_map(|chunk| {
            let chunk = chunk.as_ref().ok()?;
            (chunk.kind == INTERACTION).then(|| chunk.payload.clone())
        })
        .collect()
}

/// A settled goal is reported once, on the stream and through `drain_rail_events`.
#[tokio::test]
async fn a_settled_goal_reaches_the_stream_and_the_drain_exactly_once() -> Result<()> {
    let temp = tempfile::TempDir::new()?;
    let workspace = Workspace::for_language(temp.path(), "en");
    let store = settled_store(&workspace).await?;
    let agent = goal_agent(workspace, Arc::clone(&store))?;

    let chunks: Vec<CommonResult<AgentChunk>> = agent
        .stream(json!({"query": "finish it"}), None)
        .collect()
        .await;
    ensure!(
        chunks.iter().all(std::result::Result::is_ok),
        "the run must not fail"
    );
    let frames = interaction_frames(&chunks);
    ensure!(
        frames.len() == 1,
        "one task iteration, so exactly one event: {frames:?}"
    );
    ensure!(frames[0]["type"] == json!("goal.updated"), "{}", frames[0]);
    ensure!(
        frames[0]["payload"]["goal"]["objective"] == json!("ship the goal rail"),
        "the frame carries the persisted goal: {}",
        frames[0]
    );
    ensure!(
        chunks
            .last()
            .and_then(|chunk| chunk.as_ref().ok())
            .is_some_and(|chunk| chunk.kind == openjiuwen_agent::output::ANSWER),
        "the invocation still terminates with the answer chunk"
    );

    // The stream observes the event, and the buffer retains the same one for a host that never
    // read the stream — each event is collected exactly once.
    let drained = agent.drain_rail_events();
    ensure!(drained.len() == 1, "one retained event, got {drained:?}");
    let output = drained[0].to_output_value();
    ensure!(output["type"] == json!("goal.updated"), "{output}");
    ensure!(
        output["payload"]["goal"]["objective"] == json!("ship the goal rail"),
        "{output}"
    );
    ensure!(
        agent.drain_rail_events().is_empty(),
        "a second drain finds nothing"
    );
    Ok(())
}

/// An active goal publishes the reminder but raises no event until it settles.
#[tokio::test]
async fn an_active_goal_reminds_without_emitting() -> Result<()> {
    let temp = tempfile::TempDir::new()?;
    let workspace = Workspace::for_language(temp.path(), "en");
    let store = GoalStore::for_workspace(&workspace);
    store.start("ship the goal rail", now()).await?;

    let agent = goal_agent(workspace, Arc::clone(&store))?;
    let chunks: Vec<CommonResult<AgentChunk>> = agent
        .stream(json!({"query": "keep going"}), None)
        .collect()
        .await;
    ensure!(
        interaction_frames(&chunks).is_empty(),
        "an active goal is not a settled goal"
    );
    ensure!(
        agent.drain_rail_events().is_empty(),
        "nothing was queued for the host"
    );

    // The same rail publishes the `goal` prompt section before the iteration.
    let rail = GoalRail::new(Arc::clone(&store), GoalRailConfig::default());
    let mut ctx = HarnessRailContext::for_event(openjiuwen_agent::RailEvent::BeforeInvoke)
        .with_task(TaskIterationInfo::new(0, 0));
    rail.before_task_iteration(&mut ctx).await?;
    let sections = openjiuwen_harness::rails::prompt_sections(&ctx);
    ensure!(sections.len() == 1, "one goal section, got {sections:?}");
    ensure!(sections[0].name == "goal", "the section keeps its name");
    ensure!(
        sections[0].render("en").contains("ship the goal rail"),
        "the reminder names the objective: {}",
        sections[0].render("en")
    );
    ensure!(
        !ctx.extra().contains_key(GOAL_EVENT_EXTRA_KEY),
        "the before-iteration hook queues no event"
    );
    Ok(())
}

/// The goal rail is only auto-registered when a goal tool is mounted.
#[tokio::test]
async fn a_settled_goal_without_a_goal_tool_raises_no_event() -> Result<()> {
    let temp = tempfile::TempDir::new()?;
    let workspace = Workspace::for_language(temp.path(), "en");
    let store = settled_store(&workspace).await?;
    ensure!(store.load().await?.is_some(), "the fixture goal persists");

    let agent = DeepAgentBuilder::new(AgentCard::new("plain-agent"))
        .config(
            DeepAgentConfig::default()
                .with_language("en".to_string())
                .with_workspace(workspace)
                .with_enable_task_loop(true),
        )
        .model(Arc::new(ScriptedModel::text("done")))
        .build()?;
    let chunks: Vec<CommonResult<AgentChunk>> = agent
        .stream(json!({"query": "anything"}), None)
        .collect()
        .await;
    ensure!(
        interaction_frames(&chunks).is_empty(),
        "no goal tool, no goal rail, no event"
    );
    ensure!(
        agent.drain_rail_events().is_empty(),
        "the plain agent retains nothing"
    );
    Ok(())
}

/// The event is a plain interaction event, so it also survives a manual context round trip.
#[test]
fn a_goal_event_round_trips_through_its_output_frame() -> Result<()> {
    let event = InteractionEvent::goal_updated(Some(json!({"id": "g-1", "status": "complete"})));
    let frame = event.to_output_value();
    ensure!(frame["type"] == json!("goal.updated"), "{frame}");
    ensure!(frame["index"] == json!(0), "{frame}");
    ensure!(
        frame["payload"]["goal"] == json!({"id": "g-1", "status": "complete"}),
        "{frame}"
    );
    let on_stream = serde_json::to_value(&event)?;
    ensure!(
        on_stream["type"] == frame["type"],
        "the serde form and the stream frame agree: {on_stream} vs {frame}"
    );
    Ok(())
}
