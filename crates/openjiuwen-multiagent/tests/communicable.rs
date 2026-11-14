#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration tests for [`CommunicableAgent`].
//!
//! The suite covers the Python mixin's contract: unbound calls fail with the
//! validation error naming the agent, binding reports through `is_bound` /
//! `runtime` / `agent_id`, rebinding the same pair is a silent no-op (and a
//! different pair warns then rebinds), `unbind` drops only the runtime, and a
//! bound agent drives a real in-process runtime (`send` to a registered member,
//! `publish`/`subscribe`/`unsubscribe`). Every agent is an in-process fake and
//! no deadline exceeds the default, so the suite is offline and deterministic.

mod common;

use std::sync::Arc;

use openjiuwen_agent::{Agent, AgentOutput};
use openjiuwen_common::error::ErrorKind;
use openjiuwen_multiagent::{CommunicableAgent, RuntimeConfig, TeamRuntime};
use parking_lot::Mutex;
use serde_json::{json, Value};

use crate::common::{agent_card, provider, FakeAgent};

#[tokio::test]
async fn unbound_calls_use_the_python_validation_message() {
    let agent = CommunicableAgent::new("ghost");
    assert!(!agent.is_bound());
    assert!(agent.runtime().is_none());
    assert_eq!(agent.agent_id().as_deref(), Some("ghost"));

    let errors = [
        agent.send(json!({}), "bob", None).await.unwrap_err(),
        agent.publish(json!({}), "news", None).await.unwrap_err(),
        agent.subscribe("news").await.unwrap_err(),
        agent.unsubscribe("news").await.unwrap_err(),
    ];
    for error in errors {
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(
            error.message(),
            "agent 'ghost' is not bound to a team runtime"
        );
    }
}

#[tokio::test]
async fn bind_rebind_and_unbind_keep_the_binding_consistent() {
    let runtime = TeamRuntime::new(RuntimeConfig::new().with_team_id("binding"));
    let other = TeamRuntime::new(RuntimeConfig::new().with_team_id("other"));

    let agent = CommunicableAgent::new("alice");
    agent.bind_runtime(Arc::clone(&runtime), "alice");
    assert!(agent.is_bound());
    assert!(Arc::ptr_eq(&agent.runtime().unwrap(), &runtime));
    assert_eq!(agent.agent_id().as_deref(), Some("alice"));

    // The same `(runtime, agent_id)` pair is a silent no-op.
    agent.bind_runtime(Arc::clone(&runtime), "alice");
    assert!(Arc::ptr_eq(&agent.runtime().unwrap(), &runtime));
    assert_eq!(agent.agent_id().as_deref(), Some("alice"));

    // A different pair warns and rebinding proceeds.
    agent.bind_runtime(Arc::clone(&runtime), "alice-2");
    assert_eq!(agent.agent_id().as_deref(), Some("alice-2"));
    assert!(Arc::ptr_eq(&agent.runtime().unwrap(), &runtime));

    agent.bind_runtime(Arc::clone(&other), "alice-2");
    assert!(Arc::ptr_eq(&agent.runtime().unwrap(), &other));

    agent.unbind();
    assert!(!agent.is_bound());
    assert!(agent.runtime().is_none());
    // The identity survives unbinding, so the unbound error still names it.
    assert_eq!(agent.agent_id().as_deref(), Some("alice-2"));
    let error = agent.subscribe("news").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(
        error.message(),
        "agent 'alice-2' is not bound to a team runtime"
    );
}

#[tokio::test]
async fn bound_agent_sends_publishes_and_subscribes_through_the_runtime() {
    let runtime = TeamRuntime::new(RuntimeConfig::new().with_team_id("comm"));
    runtime.register_agent(agent_card("bob"), provider(FakeAgent::echoing("bob")));

    // `alice` is a registered member whose agent records every pub/sub payload.
    let delivered: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&delivered);
    let alice_agent: Arc<dyn Agent> =
        FakeAgent::new(agent_card("alice"), move |inputs, _session| {
            sink.lock().push(inputs.clone());
            Ok(AgentOutput::answer("seen"))
        });
    runtime.register_agent(agent_card("alice"), provider(alice_agent));
    runtime.start().await.unwrap();

    let alice = CommunicableAgent::new("alice");
    alice.bind_runtime(Arc::clone(&runtime), "alice");

    let reply = alice
        .send(json!({"query": "hi"}), "bob", Some("s1"))
        .await
        .unwrap();
    assert_eq!(reply["output"], "bob:hi");

    alice.subscribe("news").await.unwrap();
    assert_eq!(runtime.list_agent_subscriptions("alice"), vec!["news"]);
    alice.subscribe("reports.*").await.unwrap();
    assert_eq!(
        runtime.list_agent_subscriptions("alice"),
        vec!["news", "reports.*"]
    );

    alice
        .publish(json!({"headline": "hi"}), "news", None)
        .await
        .unwrap();
    assert_eq!(delivered.lock().clone(), vec![json!({"headline": "hi"})]);

    alice.unsubscribe("news").await.unwrap();
    alice.unsubscribe("reports.*").await.unwrap();
    assert_eq!(
        runtime.list_agent_subscriptions("alice"),
        Vec::<String>::new()
    );
    assert!(runtime.is_running());
    runtime.stop().await.unwrap();
}
