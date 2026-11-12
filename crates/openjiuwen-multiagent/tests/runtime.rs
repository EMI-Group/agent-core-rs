#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration tests for the [`TeamRuntime`].
//!
//! The suite registers [`common::FakeAgent`] members and drives point-to-point
//! and pub/sub deliveries through the runtime's own bus: registration order and
//! queries, per-resolution providers versus pinned instances, child-session
//! construction from a bound team session and from the envelope session id, the
//! P2P deadline, the exact validation messages, and pub/sub fan-out. Members are
//! in-process closures, so every case is offline and deterministic.

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_agent::{Agent, AgentCard, AgentChunk, AgentOutput};
use openjiuwen_common::error::{ErrorKind, Result};
use openjiuwen_multiagent::{AgentProvider, RuntimeConfig, TeamAgent, TeamRuntime};
use openjiuwen_session::{AgentTeamSession, AgentTeamSessionOptions, Session};
use parking_lot::Mutex;
use serde_json::{json, Value};

use crate::common::{agent_card, FakeAgent};

/// Builds a provider that answers with a fresh echoing [`FakeAgent`].
fn echoing_provider(name: &str) -> AgentProvider {
    let name = name.to_owned();
    Arc::new(move || TeamAgent::new(FakeAgent::echoing(&name)))
}

/// Recorded `(session id, parent id)` pairs; `None` means no session was passed.
type SeenSessions = Arc<Mutex<Vec<Option<(String, Option<String>)>>>>;

/// Builds a provider whose agent records the delivery session.
fn recording_provider(name: &str, seen: &SeenSessions) -> AgentProvider {
    let seen = Arc::clone(seen);
    let name = name.to_owned();
    Arc::new(move || {
        let seen = Arc::clone(&seen);
        let card = agent_card(&name);
        TeamAgent::new(FakeAgent::new(card, move |_inputs, session| {
            seen.lock().push(session.map(|session| {
                (
                    session.id().to_owned(),
                    session.parent_id().map(str::to_owned),
                )
            }));
            Ok(AgentOutput::answer("ok"))
        }))
    })
}

#[tokio::test]
async fn registration_and_queries() {
    let runtime = TeamRuntime::new(RuntimeConfig::new());
    assert_eq!(runtime.agent_count(), 0);
    assert_eq!(runtime.agent_ids(), Vec::<String>::new());

    runtime.register_agent(agent_card("a"), echoing_provider("a"));
    runtime.register_agent(agent_card("b"), echoing_provider("b"));

    assert_eq!(runtime.agent_ids(), vec!["a".to_string(), "b".to_string()]);
    assert_eq!(runtime.agent_count(), 2);
    assert!(runtime.has_agent("a"));
    assert!(!runtime.has_agent("c"));
    assert_eq!(
        runtime.agent_card("a").map(|card| card.name),
        Some("a".to_string())
    );
    assert!(runtime.agent_provider("a").is_some());
    assert!(runtime.agent_card("c").is_none());

    runtime.subscribe("a", "news").await.unwrap();
    assert_eq!(runtime.message_bus().subscribers("news"), vec!["a"]);

    let removed = runtime.unregister_agent("a").expect("a was registered");
    assert_eq!(removed.id, "a");
    assert_eq!(runtime.agent_ids(), vec!["b".to_string()]);
    assert_eq!(
        runtime.message_bus().subscribers("news"),
        Vec::<String>::new()
    );
    assert!(runtime.unregister_agent("a").is_none());
}

#[tokio::test]
async fn provider_is_invoked_per_resolution() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&calls);
    let provider: AgentProvider = Arc::new(move || {
        counter.fetch_add(1, Ordering::SeqCst);
        TeamAgent::new(FakeAgent::echoing("worker"))
    });
    let runtime = TeamRuntime::new(RuntimeConfig::new());
    runtime.register_agent(agent_card("worker"), provider);

    runtime
        .invoke_agent("worker", json!({"query": "one"}), None)
        .await
        .unwrap();
    runtime
        .invoke_agent("worker", json!({"query": "two"}), None)
        .await
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 2);

    let pinned = TeamAgent::new(FakeAgent::new(agent_card("worker"), |_inputs, _session| {
        Ok(AgentOutput::answer("pinned"))
    }));
    runtime.pin_agent("worker", pinned);
    let output = runtime
        .invoke_agent("worker", json!({"query": "three"}), None)
        .await
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        output.to_value(),
        json!({"output": "pinned", "result_type": "answer"})
    );
}

#[tokio::test]
async fn p2p_builds_child_session_from_bound_team_session() {
    let runtime = TeamRuntime::new(RuntimeConfig::new());
    let team_session = Arc::new(AgentTeamSession::with_options(AgentTeamSessionOptions {
        session_id: Some("s1".to_owned()),
        ..AgentTeamSessionOptions::default()
    }));
    runtime.bind_team_session(Arc::clone(&team_session));

    let seen: SeenSessions = Arc::new(Mutex::new(Vec::new()));
    runtime.register_agent(agent_card("worker"), recording_provider("worker", &seen));

    let reply = runtime
        .send(json!({"query": "hi"}), "worker", "boss", Some("s1"), None)
        .await
        .unwrap();
    assert_eq!(reply["output"], "ok");

    let recorded = seen.lock().clone();
    let (session_id, parent_id) = recorded[0].clone().expect("a child session was built");
    assert_ne!(session_id, "s1");
    assert_eq!(parent_id.as_deref(), Some("s1"));
}

#[tokio::test]
async fn p2p_without_team_session_uses_envelope_session_id() {
    let runtime = TeamRuntime::new(RuntimeConfig::new());
    let seen: SeenSessions = Arc::new(Mutex::new(Vec::new()));
    runtime.register_agent(agent_card("worker"), recording_provider("worker", &seen));

    runtime
        .send(json!({"query": "hi"}), "worker", "boss", Some("s2"), None)
        .await
        .unwrap();
    runtime
        .send(json!({"query": "hi"}), "worker", "boss", None, None)
        .await
        .unwrap();

    let recorded = seen.lock().clone();
    let (session_id, parent_id) = recorded[0].clone().expect("a child session was built");
    assert_eq!(session_id, "s2");
    assert_eq!(parent_id, None);
    assert_eq!(recorded[1], None);
}

/// An agent that sleeps before answering, to exercise the P2P deadline.
struct SlowAgent {
    card: AgentCard,
    delay: Duration,
}

#[async_trait::async_trait]
impl Agent for SlowAgent {
    fn card(&self) -> &AgentCard {
        &self.card
    }

    async fn invoke(
        &self,
        _inputs: Value,
        _session: Option<Arc<dyn Session>>,
    ) -> Result<AgentOutput> {
        tokio::time::sleep(self.delay).await;
        Ok(AgentOutput::answer("slow"))
    }

    fn stream(
        &self,
        _inputs: Value,
        _session: Option<Arc<dyn Session>>,
    ) -> BoxStream<'static, Result<AgentChunk>> {
        futures::stream::empty().boxed()
    }
}

#[tokio::test]
async fn p2p_timeout_uses_p2p_timeout() {
    let runtime = TeamRuntime::new(RuntimeConfig::new());
    let slow: Arc<dyn Agent> = Arc::new(SlowAgent {
        card: agent_card("slow"),
        delay: Duration::from_millis(200),
    });
    runtime.register_agent(
        agent_card("slow"),
        Arc::new(move || TeamAgent::new(Arc::clone(&slow))),
    );

    runtime.set_p2p_timeout(Duration::from_millis(20));
    let error = runtime
        .send(json!({}), "slow", "boss", None, None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Timeout);
    assert!(error.message().contains("timed out after"), "{error}");

    let reply = runtime
        .send(
            json!({}),
            "slow",
            "boss",
            None,
            Some(Duration::from_secs(5)),
        )
        .await
        .unwrap();
    assert_eq!(reply["output"], "slow");
}

#[tokio::test]
async fn send_validation_messages() {
    let runtime = TeamRuntime::new(RuntimeConfig::new());
    runtime.register_agent(agent_card("worker"), echoing_provider("worker"));

    let error = runtime
        .send(json!({}), "worker", "", None, None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(error.message(), "sender is required for message tracing");

    let error = runtime
        .send(json!({}), "", "boss", None, None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(error.message(), "recipient is required");

    let error = runtime
        .send(json!({}), "nobody", "boss", None, None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(
        error.message(),
        "Recipient 'nobody' not registered in runtime"
    );

    let error = runtime
        .publish(json!({}), "news", "", None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(error.message(), "sender is required for message tracing");

    let error = runtime.subscribe("", "news").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(error.message(), "agent_id is required");

    let error = runtime.subscribe("a", "").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(error.message(), "topic is required");
}

#[tokio::test]
async fn publish_delivers_to_subscribers() {
    let runtime = TeamRuntime::new(RuntimeConfig::new());
    let delivered: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    for name in ["a", "b"] {
        let delivered = Arc::clone(&delivered);
        let label = name.to_owned();
        let card = agent_card(name);
        let agent: Arc<dyn Agent> = FakeAgent::new(card, move |_inputs, _session| {
            delivered.lock().push(label.clone());
            Ok(AgentOutput::answer("seen"))
        });
        runtime.register_agent(
            agent_card(name),
            Arc::new(move || TeamAgent::new(Arc::clone(&agent))),
        );
    }

    runtime.subscribe("a", "news").await.unwrap();
    runtime.subscribe("b", "news").await.unwrap();
    runtime
        .publish(json!({"headline": "hi"}), "news", "boss", None)
        .await
        .unwrap();

    let mut seen = delivered.lock().clone();
    seen.sort();
    assert_eq!(seen, vec!["a".to_string(), "b".to_string()]);
}
