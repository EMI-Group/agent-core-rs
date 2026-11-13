#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration tests for the P2P ability manager.
//!
//! Every member is an in-process [`AsyncFakeAgent`], every batch runs through a
//! real [`TeamRuntime`] (so dispatch goes over the crate's message bus), and no
//! test needs a model, a network, or a credential. Shared counters and vectors
//! are `std` atomics and mutexes held only for the mutation itself.

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::future::BoxFuture;
use futures::FutureExt;
use openjiuwen_agent::{Agent, AgentCard, AgentOutput};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_multiagent::ability::{
    AbilityCall, AbilityExecutor, AbilityRoute, P2PAbilityManager,
};
use openjiuwen_multiagent::{RuntimeConfig, TeamRuntime};
use openjiuwen_session::{AgentSession, AgentSessionOptions, Session};
use serde_json::{json, Value};

use crate::common::{agent_card, provider, AsyncFakeAgent};

/// The invocation handler a test member is built from.
type MemberHandler = Box<
    dyn Fn(Value, Option<Arc<dyn Session>>) -> BoxFuture<'static, Result<AgentOutput>>
        + Send
        + Sync,
>;

/// Registers `card` on `runtime` with an invocation driven by `handler`.
fn register(runtime: &Arc<TeamRuntime>, card: AgentCard, handler: MemberHandler) {
    let agent = AsyncFakeAgent::new(card.clone(), handler);
    runtime.register_agent(card, provider(Arc::clone(&agent) as Arc<dyn Agent>));
}

/// Registers a member that always answers `text`.
fn register_answer(runtime: &Arc<TeamRuntime>, name: &str, text: &str) {
    let answer = text.to_owned();
    register(
        runtime,
        agent_card(name),
        Box::new(move |_inputs, _session| {
            let answer = answer.clone();
            async move { Ok(AgentOutput::answer(answer)) }.boxed()
        }),
    );
}

/// A local executor echoing the call name and its arguments.
struct EchoExecutor;

#[async_trait]
impl AbilityExecutor for EchoExecutor {
    async fn execute(
        &self,
        call: &AbilityCall,
        _session: Option<Arc<dyn Session>>,
    ) -> Result<Value> {
        Ok(json!({"local": call.name, "arguments": call.arguments}))
    }
}

/// A local executor that records the name of every call it serves.
struct RecordingExecutor {
    /// Completion order, appended on every call.
    order: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl AbilityExecutor for RecordingExecutor {
    async fn execute(
        &self,
        call: &AbilityCall,
        _session: Option<Arc<dyn Session>>,
    ) -> Result<Value> {
        self.order.lock().unwrap().push(call.name.clone());
        Ok(json!({"local": call.name}))
    }
}

/// A local executor that always fails.
struct FailingExecutor;

#[async_trait]
impl AbilityExecutor for FailingExecutor {
    async fn execute(
        &self,
        _call: &AbilityCall,
        _session: Option<Arc<dyn Session>>,
    ) -> Result<Value> {
        Err(Error::tool("local tool exploded"))
    }
}

#[tokio::test]
async fn results_follow_the_input_order() {
    let runtime = TeamRuntime::new(RuntimeConfig::new().with_team_id("ability"));
    register_answer(&runtime, "alpha", "a");
    register_answer(&runtime, "beta", "b");
    register_answer(&runtime, "gamma", "c");
    let manager = P2PAbilityManager::new(Arc::clone(&runtime), "supervisor")
        .with_executor(Arc::new(EchoExecutor));
    for name in ["alpha", "beta", "gamma"] {
        manager.register_agent_tool(&agent_card(name));
    }
    assert_eq!(manager.agent_tool_ids(), vec!["alpha", "beta", "gamma"]);
    assert!(!manager.is_agent_tool("local-tool"));

    let calls = vec![
        AbilityCall::new("alpha", json!({"n": 1})).with_call_id("c1"),
        AbilityCall::new("local-tool", json!({"n": 2})).with_call_id("c2"),
        AbilityCall::new("beta", json!({"n": 3})).with_call_id("c3"),
        AbilityCall::new("gamma", json!({"n": 4})).with_call_id("c4"),
    ];
    let results = manager.execute(calls, None).await.unwrap();
    let ids: Vec<&str> = results
        .iter()
        .map(|result| result.call_id.as_str())
        .collect();
    assert_eq!(ids, ["c1", "c2", "c3", "c4"]);
    let routes: Vec<AbilityRoute> = results.iter().map(|result| result.route).collect();
    assert_eq!(
        routes,
        [
            AbilityRoute::P2p,
            AbilityRoute::Local,
            AbilityRoute::P2p,
            AbilityRoute::P2p,
        ]
    );
    assert_eq!(results[0].output["output"], "a");
    assert_eq!(results[1].output["local"], "local-tool");
    assert_eq!(results[1].output["arguments"], json!({"n": 2}));
    assert_eq!(results[2].output["output"], "b");
    assert_eq!(results[3].output["output"], "c");
    assert!(results.iter().all(|result| !result.is_error()));
}

#[tokio::test]
async fn a_failing_agent_call_does_not_abort_the_batch() {
    let runtime = TeamRuntime::new(RuntimeConfig::new().with_team_id("ability"));
    register_answer(&runtime, "good", "fine");
    register(
        &runtime,
        agent_card("bad"),
        Box::new(|_inputs, _session| async move { Err(Error::validation("p2p exploded")) }.boxed()),
    );
    let manager = P2PAbilityManager::new(Arc::clone(&runtime), "supervisor");
    for name in ["good", "bad"] {
        manager.register_agent_tool(&agent_card(name));
    }

    let calls = vec![
        AbilityCall::new("good", json!({})).with_call_id("c1"),
        AbilityCall::new("bad", json!({})).with_call_id("c2"),
    ];
    let results = manager.execute(calls, None).await.unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].route, AbilityRoute::P2p);
    assert_eq!(results[0].output["output"], "fine");
    assert_eq!(results[1].route, AbilityRoute::Error);
    assert!(results[1].is_error());
    let message = results[1].error.as_deref().unwrap();
    assert!(
        message.starts_with("P2P dispatch to 'bad' failed:"),
        "{message}"
    );
    assert!(message.contains("p2p exploded"), "{message}");
}

#[tokio::test]
async fn an_empty_supervisor_id_cannot_dispatch() {
    let runtime = TeamRuntime::new(RuntimeConfig::new().with_team_id("ability"));
    register_answer(&runtime, "alpha", "a");
    // The manager's supervisor id is the P2P `sender`; the runtime rejects an
    // empty one, which surfaces as a per-call error instead of a silent drop.
    let manager = P2PAbilityManager::new(runtime, "");
    manager.register_agent_tool(&agent_card("alpha"));
    let results = manager
        .execute(vec![AbilityCall::new("alpha", json!({}))], None)
        .await
        .unwrap();
    assert_eq!(results[0].route, AbilityRoute::Error);
    let message = results[0].error.as_deref().unwrap();
    assert!(message.contains("sender is required"), "{message}");
}

#[tokio::test]
async fn local_failures_keep_the_executor_message() {
    let runtime = TeamRuntime::new(RuntimeConfig::new().with_team_id("ability"));
    let manager =
        P2PAbilityManager::new(runtime, "supervisor").with_executor(Arc::new(FailingExecutor));
    let results = manager
        .execute(vec![AbilityCall::new("any-tool", json!({}))], None)
        .await
        .unwrap();
    assert_eq!(results[0].route, AbilityRoute::Error);
    assert_eq!(results[0].error.as_deref(), Some("local tool exploded"));
    assert!(results[0].output.is_null());
}

#[tokio::test]
async fn a_local_call_without_an_executor_is_an_explicit_error() {
    let runtime = TeamRuntime::new(RuntimeConfig::new().with_team_id("ability"));
    let manager = P2PAbilityManager::new(runtime, "supervisor");
    let results = manager
        .execute(vec![AbilityCall::new("unknown-tool", json!({}))], None)
        .await
        .unwrap();
    assert_eq!(results[0].route, AbilityRoute::Error);
    let message = results[0].error.as_deref().unwrap();
    assert!(
        message.contains("no ability executor configured"),
        "{message}"
    );
}

#[tokio::test]
async fn the_parallel_cap_bounds_concurrent_agent_calls() {
    let runtime = TeamRuntime::new(RuntimeConfig::new().with_team_id("ability"));
    let current = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    for name in ["a", "b", "c", "d"] {
        let current = Arc::clone(&current);
        let peak = Arc::clone(&peak);
        register(
            &runtime,
            agent_card(name),
            Box::new(move |_inputs, _session| {
                let current = Arc::clone(&current);
                let peak = Arc::clone(&peak);
                async move {
                    let in_flight = current.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(in_flight, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    current.fetch_sub(1, Ordering::SeqCst);
                    Ok(AgentOutput::answer("done"))
                }
                .boxed()
            }),
        );
    }
    let manager = P2PAbilityManager::new(Arc::clone(&runtime), "supervisor").with_max_parallel(2);
    for name in ["a", "b", "c", "d"] {
        manager.register_agent_tool(&agent_card(name));
    }

    let calls: Vec<AbilityCall> = ["a", "b", "c", "d"]
        .iter()
        .map(|name| AbilityCall::new(*name, json!({})))
        .collect();
    let results = manager.execute(calls, None).await.unwrap();
    assert_eq!(results.len(), 4);
    assert!(results
        .iter()
        .all(|result| result.route == AbilityRoute::P2p && !result.is_error()));
    // The bus serializes same-topic P2P deliveries, so the peak can be lower
    // than the cap; the assertion pins what the manager enforces.
    let observed = peak.load(Ordering::SeqCst);
    assert!(
        observed <= 2,
        "peak concurrency {observed} exceeded the cap"
    );
    assert_eq!(current.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn local_calls_do_not_wait_for_agent_permits() {
    let runtime = TeamRuntime::new(RuntimeConfig::new().with_team_id("ability"));
    let order = Arc::new(Mutex::new(Vec::new()));
    let agent_order = Arc::clone(&order);
    register(
        &runtime,
        agent_card("slow"),
        Box::new(move |_inputs, _session| {
            let order = Arc::clone(&agent_order);
            async move {
                tokio::time::sleep(Duration::from_millis(150)).await;
                order.lock().unwrap().push("agent".to_owned());
                Ok(AgentOutput::answer("slow"))
            }
            .boxed()
        }),
    );
    let manager = P2PAbilityManager::new(Arc::clone(&runtime), "supervisor")
        .with_max_parallel(1)
        .with_executor(Arc::new(RecordingExecutor {
            order: Arc::clone(&order),
        }));
    manager.register_agent_tool(&agent_card("slow"));

    let calls = vec![
        AbilityCall::new("slow", json!({})),
        AbilityCall::new("quick", json!({})),
    ];
    let results = manager.execute(calls, None).await.unwrap();
    assert_eq!(results[0].route, AbilityRoute::P2p);
    assert_eq!(results[1].route, AbilityRoute::Local);
    // The single permit belongs to the agent call, so a local call finishing
    // first proves it never queued behind an agent dispatch.
    assert_eq!(
        *order.lock().unwrap(),
        vec!["quick".to_owned(), "agent".to_owned()]
    );
}

#[tokio::test]
async fn the_call_session_id_reaches_the_member() {
    let runtime = TeamRuntime::new(RuntimeConfig::new().with_team_id("ability"));
    let seen: Arc<Mutex<Vec<Option<String>>>> = Arc::new(Mutex::new(Vec::new()));
    let recorder = Arc::clone(&seen);
    register(
        &runtime,
        agent_card("seen"),
        Box::new(move |_inputs, session| {
            let recorder = Arc::clone(&recorder);
            async move {
                recorder
                    .lock()
                    .unwrap()
                    .push(session.map(|session| session.id().to_owned()));
                Ok(AgentOutput::answer("ok"))
            }
            .boxed()
        }),
    );
    let manager = P2PAbilityManager::new(runtime, "supervisor");
    manager.register_agent_tool(&agent_card("seen"));

    let session: Arc<dyn Session> = Arc::new(AgentSession::with_options(AgentSessionOptions {
        session_id: Some("s-42".to_owned()),
        ..AgentSessionOptions::default()
    }));
    let results = manager
        .execute(vec![AbilityCall::new("seen", json!({}))], Some(session))
        .await
        .unwrap();
    assert_eq!(results[0].route, AbilityRoute::P2p);
    assert_eq!(seen.lock().unwrap().as_slice(), [Some("s-42".to_owned())]);
}

#[tokio::test]
async fn an_empty_batch_returns_no_results() {
    let runtime = TeamRuntime::new(RuntimeConfig::new().with_team_id("ability"));
    let manager = P2PAbilityManager::new(runtime, "supervisor");
    assert_eq!(manager.execute(Vec::new(), None).await.unwrap().len(), 0);
}
