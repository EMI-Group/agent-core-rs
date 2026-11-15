//! Enforcement of [`RunnerConfig::distributed_mode`] on every entry point that
//! would need a distributed transport.
//!
//! The three execution entry points of each shape (`run_agent`,
//! `run_workflow`, `run_agent_team`), their streaming variants, the three pool
//! admission paths, and [`Runner::start`] all consult **one** guard
//! (`RunnerConfig::require_local`, reached through `RunServices::require_local`),
//! so a distributed-configured runner refuses with the same typed
//! `Error::Unsupported` — naming the mode and the deferred transports (`Pulsar`,
//! `ZeroMQ`, `A2A` reply-topic) — instead of running in-process under a distributed
//! name.
//!
//! The spawn seam is deliberately exempt: a spawned child is selected by its
//! backend, not by the runner's transport mode, so a local child keeps working on
//! a distributed-configured runner while `remote_mq`/`remote_a2a` stay typed
//! `Unsupported`.
//!
//! Every case is offline and bounded: in-process scripted agents, the real
//! `BaseTeam`/handoff fixture, and a single in-process spawn handler.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_agent::output::{AgentChunk, AgentOutput};
use openjiuwen_agent::Agent;
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_multiagent::Team;
use openjiuwen_runner::callback::CallbackFramework;
use openjiuwen_runner::config::RunnerConfig;
use openjiuwen_runner::monitor::RunPhase;
use openjiuwen_runner::resources::ResourceMgr;
use openjiuwen_runner::run::{
    invoke_agent, invoke_team, invoke_workflow, stream_agent, stream_team, stream_workflow,
    AgentRef, RunOptions, RunServices, TeamRef, WorkflowRef,
};
use openjiuwen_runner::{
    AgentSpawner, InProcessHandler, InProcessSpawner, Runner, SpawnBackend, SpawnSpec,
    SpawnerRegistry,
};
use openjiuwen_workflow::WorkflowCard;
use serde_json::{json, Value};

use common::{
    agent_card, handoff_team_fixture, passthrough_workflow, team_calls, team_card, ScriptedAgent,
    TeamCalls, HANDOFF_SECOND_ANSWER,
};

/// Upper bound on a deterministic in-memory wait (a deadline, never an ordering
/// device).
const BOUND: Duration = Duration::from_secs(10);

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// A runner configured for distributed mode through the documented builder.
fn distributed() -> Arc<Runner> {
    Runner::with_config(RunnerConfig::default().with_distributed_mode(true))
}

/// The execution context a direct `run::*` call would receive from a
/// distributed-configured runner.
fn distributed_services() -> RunServices {
    RunServices::new(
        ResourceMgr::new(),
        CallbackFramework::new(),
        RunnerConfig::default().with_distributed_mode(true),
        None,
    )
}

/// Registers one agent, one workflow, and one handoff team on `runner`, and
/// returns the team's call log so a suite can prove nothing ran.
fn register_fixtures(runner: &Arc<Runner>) -> Arc<TeamCalls> {
    let resources = runner.resource_mgr();
    resources
        .register_agent(
            agent_card("assistant"),
            Arc::new(|| {
                Arc::new(ScriptedAgent::new("assistant").with_answer("local answer"))
                    as Arc<dyn Agent>
            }),
        )
        .expect("the agent registers");
    resources
        .register_workflow(
            WorkflowCard::new("wf"),
            Arc::new(|| passthrough_workflow("wf")),
        )
        .expect("the workflow registers");
    let calls = team_calls();
    let team = handoff_team_fixture("crew", &calls);
    resources
        .register_team(team_card("crew"), {
            let team = Arc::clone(&team);
            Arc::new(move || Arc::clone(&team) as Arc<dyn Team>)
        })
        .expect("the team registers");
    calls
}

/// Asserts `error` is the shared distributed-mode refusal.
fn assert_refusal(error: &Error) {
    assert_eq!(
        error.kind(),
        ErrorKind::Unsupported,
        "a distributed-configured runner refuses with Unsupported: {error}"
    );
    let message = error.to_string();
    assert!(
        message.contains("distributed runner mode"),
        "`{message}` should name the mode"
    );
    for transport in ["Pulsar", "ZeroMQ", "A2A"] {
        assert!(
            message.contains(transport),
            "`{message}` should name the deferred {transport} transport"
        );
    }
}

/// Drains `stream`, asserting it holds exactly one frame: the refusal.
async fn only_refusal_frame<T>(mut stream: BoxStream<'static, Result<T>>) -> Error {
    let error = match stream.next().await {
        Some(Err(error)) => error,
        Some(Ok(_)) => panic!("the refusal frame is an error"),
        None => panic!("the refusal is the stream's one frame"),
    };
    assert!(
        stream.next().await.is_none(),
        "nothing follows the refusal frame"
    );
    error
}

/// Asserts nothing was admitted anywhere on `runner`.
fn assert_nothing_admitted(runner: &Arc<Runner>, label: &str) {
    assert!(
        runner.list_active_teams().is_empty(),
        "{label}: no pooled run is admitted"
    );
    assert!(
        runner.monitor().list_active().is_empty(),
        "{label}: no run status is begun"
    );
    let resources = runner.resource_mgr();
    assert_eq!(
        resources.agents().len(),
        1,
        "{label}: registries are intact"
    );
    assert_eq!(
        resources.workflows().len(),
        1,
        "{label}: registries are intact"
    );
    assert_eq!(resources.teams().len(), 1, "{label}: registries are intact");
}

// ---------------------------------------------------------------------------
// Invoke entry points
// ---------------------------------------------------------------------------

/// The three `invoke_*` paths and the three pool admission paths all refuse a
/// distributed-configured runner before resolving, leasing, queueing, or
/// observing anything.
#[tokio::test]
async fn every_invoke_and_pool_entry_point_refuses_a_distributed_runner() {
    let runner = distributed();
    let calls = register_fixtures(&runner);

    let error = runner
        .run_agent("assistant", json!({"query": "hi"}), RunOptions::default())
        .await
        .map(|_| ())
        .expect_err("run_agent refuses");
    assert_refusal(&error);

    let error = runner
        .run_workflow("wf", json!({"query": "hi"}), RunOptions::default())
        .await
        .map(|_| ())
        .expect_err("run_workflow refuses");
    assert_refusal(&error);

    let error = runner
        .run_agent_team(
            "crew",
            json!({"query": "hi", "conversation_id": "conv-direct"}),
            RunOptions::default(),
        )
        .await
        .map(|_| ())
        .expect_err("run_agent_team refuses");
    assert_refusal(&error);

    let inputs = json!({"query": "hi", "conversation_id": "conv-pool"});
    let error = runner
        .start_agent_team("crew", inputs.clone(), RunOptions::default())
        .await
        .map(|_| ())
        .expect_err("start_agent_team refuses");
    assert_refusal(&error);

    let error = runner
        .try_start_agent_team("crew", inputs.clone(), RunOptions::default())
        .await
        .map(|_| ())
        .expect_err("try_start_agent_team refuses");
    assert_refusal(&error);

    let error = runner
        .acquire_agent_team("crew", inputs, RunOptions::default())
        .await
        .map(|_| ())
        .expect_err("acquire_agent_team refuses");
    assert_refusal(&error);

    assert_nothing_admitted(&runner, "after every refusal");
    assert!(
        calls.inputs().is_empty(),
        "no team member ran: the team never started"
    );
}

/// Streaming refusals are the stream's **single** terminal `Err` frame: nothing
/// else happens, so no run status or hub entry can appear.
#[tokio::test]
async fn every_streaming_entry_point_refuses_with_one_terminal_frame() {
    let runner = distributed();
    let calls = register_fixtures(&runner);

    let error = only_refusal_frame(runner.run_agent_streaming(
        "assistant",
        json!({"query": "hi"}),
        RunOptions::default(),
    ))
    .await;
    assert_refusal(&error);

    let error = only_refusal_frame(runner.run_workflow_streaming(
        "wf",
        json!({"query": "hi"}),
        RunOptions::default(),
    ))
    .await;
    assert_refusal(&error);

    let error = only_refusal_frame(runner.run_agent_team_streaming(
        "crew",
        json!({"query": "hi", "conversation_id": "conv-stream"}),
        RunOptions::default(),
    ))
    .await;
    assert_refusal(&error);

    assert_nothing_admitted(&runner, "after the streaming refusals");
    assert!(
        calls.inputs().is_empty(),
        "a refused stream never starts the team"
    );
}

/// The guard lives on `RunServices`, so the free functions refuse too — the
/// facade is not where the enforcement lives.
#[tokio::test]
async fn the_free_functions_share_the_guard() {
    let services = distributed_services();

    let agent: Arc<dyn Agent> = Arc::new(ScriptedAgent::new("assistant").with_answer("unused"));
    let error = invoke_agent(
        services.clone(),
        AgentRef::of(agent),
        json!({"query": "hi"}),
        RunOptions::default(),
    )
    .await
    .map(|_| ())
    .expect_err("invoke_agent refuses");
    assert_refusal(&error);

    let agent: Arc<dyn Agent> = Arc::new(ScriptedAgent::new("assistant").with_answer("unused"));
    let error = only_refusal_frame(stream_agent(
        services.clone(),
        AgentRef::of(agent),
        json!({"query": "hi"}),
        RunOptions::default(),
    ))
    .await;
    assert_refusal(&error);

    let error = invoke_workflow(
        services.clone(),
        WorkflowRef::of(Arc::new(passthrough_workflow("wf"))),
        json!({"query": "hi"}),
        RunOptions::default(),
    )
    .await
    .map(|_| ())
    .expect_err("invoke_workflow refuses");
    assert_refusal(&error);

    let error = only_refusal_frame(stream_workflow(
        services.clone(),
        WorkflowRef::of(Arc::new(passthrough_workflow("wf"))),
        json!({"query": "hi"}),
        RunOptions::default(),
    ))
    .await;
    assert_refusal(&error);

    let calls = team_calls();
    let team = handoff_team_fixture("crew", &calls);
    let error = invoke_team(
        services.clone(),
        TeamRef::of(Arc::clone(&team) as Arc<dyn Team>),
        json!({"query": "hi"}),
        RunOptions::default(),
    )
    .await
    .map(|_| ())
    .expect_err("invoke_team refuses");
    assert_refusal(&error);

    let error = only_refusal_frame(stream_team(
        services,
        TeamRef::of(team as Arc<dyn Team>),
        json!({"query": "hi"}),
        RunOptions::default(),
    ))
    .await;
    assert_refusal(&error);

    assert!(
        calls.inputs().is_empty(),
        "a refused direct run never invokes the team"
    );
}

/// `Runner::start` keeps its own refusal, now with the shared wording.
#[tokio::test]
async fn start_still_refuses_a_distributed_runner() {
    let runner = distributed();
    assert!(runner.config().distributed_mode, "the flag is preserved");
    let error = runner.start().await.expect_err("start refuses");
    assert_refusal(&error);
    assert!(!runner.is_started(), "the runner stays stopped");
}

// ---------------------------------------------------------------------------
// Spawn: backend-selected, not mode-selected
// ---------------------------------------------------------------------------

/// The smallest hosted handler the in-process backend accepts.
struct EchoHandler {
    /// Whether the handler still serves calls.
    alive: AtomicBool,
}

impl EchoHandler {
    /// An alive handler.
    fn new() -> Arc<Self> {
        Arc::new(Self {
            alive: AtomicBool::new(true),
        })
    }
}

#[async_trait]
#[allow(clippy::unused_async_trait_impl)]
impl InProcessHandler for EchoHandler {
    fn is_alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }

    async fn send(&self, inputs: Value) -> Result<AgentOutput> {
        Ok(AgentOutput::answer(format!("echo: {inputs}")))
    }

    fn stream(&self, _inputs: Value) -> BoxStream<'static, Result<AgentChunk>> {
        Box::pin(futures::stream::empty())
    }

    async fn shutdown(&self) -> Result<()> {
        self.alive.store(false, Ordering::SeqCst);
        Ok(())
    }
}

/// `spawner` as the trait object the runner takes.
fn as_spawner<S: AgentSpawner + 'static>(spawner: Arc<S>) -> Arc<dyn AgentSpawner> {
    spawner
}

/// A local child needs no distributed transport, so a distributed-configured
/// runner still spawns and drives it end to end.
#[tokio::test]
async fn spawning_a_local_child_works_on_a_distributed_runner() {
    let runner = distributed();
    runner.set_spawner(as_spawner(Arc::new(InProcessSpawner::from_handler(
        EchoHandler::new() as Arc<dyn InProcessHandler>,
    ))));
    assert_eq!(
        runner.spawn_backend(),
        SpawnBackend::InProcess,
        "the local backend is installed"
    );

    let child = runner
        .spawn_agent(SpawnSpec::default())
        .await
        .expect("a local child needs no distributed transport");
    let output = child
        .send(json!({"ping": 1}))
        .await
        .expect("the child answers");
    assert_eq!(output.output, "echo: {\"ping\":1}");
    assert!(
        runner.active_spawns().len() == 1,
        "the live child is observed as a spawn run"
    );

    child
        .shutdown(Duration::from_secs(1))
        .await
        .expect("the child shuts down");
    assert!(
        runner.active_spawns().is_empty(),
        "an observed shutdown finishes the spawn run"
    );
}

/// A deferred transport is still a typed refusal on a distributed-configured
/// runner: the exemption is for local backends only.
#[tokio::test]
async fn a_deferred_spawn_backend_still_refuses_on_a_distributed_runner() {
    let runner = distributed();
    runner.set_spawner_registry(Arc::new(SpawnerRegistry::with_deferred()));

    let error = runner
        .spawn_agent_with(SpawnBackend::RemoteMq, SpawnSpec::default())
        .await
        .map(|_| ())
        .expect_err("the MQ transport is deferred");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(
        error.to_string().contains("Pulsar"),
        "the deferral names its transport: {error}"
    );
    assert!(
        runner.active_spawns().is_empty(),
        "a deferred backend spawns nothing"
    );
}

// ---------------------------------------------------------------------------
// Regression: a local runner still runs everything
// ---------------------------------------------------------------------------

/// Nothing about the guard changes a local runner: agent, workflow, direct team,
/// and pooled team runs all complete.
#[tokio::test]
async fn a_local_runner_still_runs_every_entry_point() {
    let runner = Runner::new();
    let calls = register_fixtures(&runner);

    let output = runner
        .run_agent("assistant", json!({"query": "hi"}), RunOptions::default())
        .await
        .expect("the agent runs");
    assert_eq!(output.output, "local answer");

    let output = runner
        .run_workflow("wf", json!({"query": "hi"}), RunOptions::default())
        .await
        .expect("the workflow runs");
    assert_eq!(output.result, json!({"output": {"answer": "hi"}}));

    let result = runner
        .run_agent_team(
            "crew",
            json!({"query": "hi", "conversation_id": "conv-direct"}),
            RunOptions::default(),
        )
        .await
        .expect("the direct team run completes");
    assert_eq!(result["output"], json!(HANDOFF_SECOND_ANSWER));

    let run = runner
        .start_agent_team(
            "crew",
            json!({"query": "hi", "conversation_id": "conv-pooled"}),
            RunOptions::default(),
        )
        .await
        .expect("the pooled team run is admitted");
    let pooled = tokio::time::timeout(BOUND, run.wait())
        .await
        .expect("the pooled run finishes before the deadline")
        .expect("the pooled run completes");
    // `wait` hands back the last raw `stream_team` frame: the transport envelope,
    // not `invoke_team`'s unwrapped answer.
    assert_eq!(pooled["payload"]["output"], json!(HANDOFF_SECOND_ANSWER));
    assert_eq!(
        runner
            .agent_team_status("crew", "conv-pooled")
            .map(|status| status.phase),
        Some(RunPhase::Completed),
        "the pooled run is mirrored as completed"
    );
    assert!(!calls.inputs().is_empty(), "the team members really ran");
}
