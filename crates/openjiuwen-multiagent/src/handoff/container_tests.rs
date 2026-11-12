use std::sync::Arc;
use std::time::Duration;

use futures::{FutureExt, StreamExt};
use openjiuwen_agent::{AgentCard, Observation, ReActAgent, ResultType};
use openjiuwen_common::error::{Error, ErrorKind};
use openjiuwen_session::{AgentTeamSession, AgentTeamSessionOptions, Session, StreamEvent};
use parking_lot::Mutex;
use serde_json::{json, Value};

use super::*;
use crate::handoff::support::{agent_card, fake_provider, FakeAgent};
use crate::handoff::{HandoffConfig, HandoffRoute, COORDINATOR_STATE_KEY};
use crate::runtime::RuntimeConfig;
use crate::{AgentProvider, TeamAgent};

/// One container plus the team session and coordinator it drives.
struct Fixture {
    runtime: Arc<TeamRuntime>,
    team_session: Arc<AgentTeamSession>,
    coordinator: Arc<HandoffOrchestrator>,
    registry: Arc<HandoffRegistry>,
}

impl Fixture {
    fn new(members: &[&str], session_id: &str, config: &HandoffConfig) -> Self {
        let runtime = TeamRuntime::new(RuntimeConfig::new().with_team_id("container-tests"));
        let team_session = Arc::new(AgentTeamSession::with_options(AgentTeamSessionOptions {
            session_id: Some(session_id.to_owned()),
            team_id: Some("container-team".to_owned()),
            ..AgentTeamSessionOptions::default()
        }));
        runtime.bind_team_session(Arc::clone(&team_session));
        let ids: Vec<String> = members.iter().map(|id| (*id).to_owned()).collect();
        let coordinator = Arc::new(HandoffOrchestrator::new(members[0], &ids, config));
        let registry = Arc::new(HandoffRegistry::new());
        registry.insert(session_id, Arc::clone(&coordinator));
        Self {
            runtime,
            team_session,
            coordinator,
            registry,
        }
    }

    /// The child session the runtime hands to the endpoint.
    fn child(&self) -> Arc<dyn Session> {
        Arc::new(self.team_session.create_agent_session("endpoint"))
    }

    fn container(&self, target: &str, allowed: &[&str], provider: AgentProvider) -> ContainerAgent {
        ContainerAgent::new(
            agent_card(target),
            provider,
            allowed.iter().map(|id| (*id).to_owned()).collect(),
            "team-card".to_owned(),
            format!("__handoff_ep_team-card_{target}"),
            Arc::downgrade(&self.runtime),
            Arc::clone(&self.registry),
        )
    }
}

fn request(input: Value, history: Vec<HandoffHistoryEntry>) -> Value {
    HandoffRequest::new(input, history).to_value()
}

#[tokio::test]
async fn missing_session_reports_an_external_error() {
    let fixture = Fixture::new(&["a"], "no-session", &HandoffConfig::default());
    let container = fixture.container("a", &[], fake_provider(FakeAgent::echo("a")));

    let error = container
        .invoke(request(json!({"query": "hi"}), vec![]), None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::External);
    assert!(
        error.message().contains("session_id is empty"),
        "unexpected message: {}",
        error.message()
    );
}

#[tokio::test]
async fn missing_coordinator_reports_an_external_error() {
    let fixture = Fixture::new(&["a"], "no-coordinator", &HandoffConfig::default());
    fixture.registry.remove("no-coordinator");
    let container = fixture.container("a", &[], fake_provider(FakeAgent::echo("a")));
    let child = fixture.child();

    let error = container
        .invoke(request(json!({"query": "hi"}), vec![]), Some(child))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::External);
    assert!(
        error.message().contains("coordinator not found"),
        "unexpected message: {}",
        error.message()
    );
    assert!(error.message().contains("no-coordinator"));
}

#[tokio::test]
async fn a_hop_writes_its_result_and_completes_the_chain() {
    let fixture = Fixture::new(&["a"], "hop-ok", &HandoffConfig::default());
    let container = fixture.container("a", &[], fake_provider(FakeAgent::echo("a")));
    let child = fixture.child();
    let mut events = fixture.team_session.subscribe();

    let output = container
        .invoke(request(json!({"query": "hi"}), vec![]), Some(child))
        .await
        .unwrap();
    assert_eq!(output.output, "{}");
    assert_eq!(output.result_type, ResultType::Answer);

    let event = events.next().await.unwrap().unwrap();
    let StreamEvent::Item(item) = event else {
        panic!("expected a stream item, got {event:?}");
    };
    let wire = item.to_wire();
    assert_eq!(wire["type"], "message");
    assert_eq!(wire["index"], 0);
    assert_eq!(wire["payload"]["output"], "a:hi");

    let result = fixture.coordinator.wait(None).await.unwrap();
    assert_eq!(result["output"], "a:hi");
}

#[tokio::test]
async fn observations_advance_the_coordinator() {
    let fixture = Fixture::new(&["a", "b"], "hop-next", &HandoffConfig::default());
    let agent = FakeAgent::sync(agent_card("a"), |_inputs, _session| {
        Ok(AgentOutput {
            output: "a is done".to_owned(),
            observations: vec![Observation {
                content: json!({"__handoff_to__": "b", "__handoff_reason__": "needs b"})
                    .to_string(),
                ..Observation::default()
            }],
            ..AgentOutput::default()
        })
    });
    let container = fixture.container("a", &["b"], fake_provider(agent));
    let child = fixture.child();

    let output = container
        .invoke(request(json!({"query": "hi"}), vec![]), Some(child))
        .await
        .unwrap();
    assert_eq!(output.output, "{}");
    assert_eq!(fixture.coordinator.current_agent_id(), "b");
    assert_eq!(fixture.coordinator.handoff_count(), 1);

    // The next hop is published from a spawned task; the chain stays
    // unresolved, so the deadline elapses.
    let error = fixture
        .coordinator
        .wait(Some(Duration::from_millis(20)))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Timeout);
}

#[tokio::test]
async fn a_blocked_handoff_completes_the_chain() {
    let config = HandoffConfig {
        routes: vec![HandoffRoute::new("b", "a")],
        ..HandoffConfig::default()
    };
    let fixture = Fixture::new(&["a", "b"], "hop-blocked", &config);
    let agent = FakeAgent::sync(agent_card("a"), |_inputs, _session| {
        Ok(AgentOutput {
            output: "a is done".to_owned(),
            observations: vec![Observation {
                content: json!({"__handoff_to__": "b"}).to_string(),
                ..Observation::default()
            }],
            ..AgentOutput::default()
        })
    });
    let container = fixture.container("a", &[], fake_provider(agent));
    let child = fixture.child();

    container
        .invoke(request(json!({"query": "hi"}), vec![]), Some(child))
        .await
        .unwrap();
    assert_eq!(fixture.coordinator.current_agent_id(), "a");
    assert_eq!(fixture.coordinator.handoff_count(), 0);
    let result = fixture.coordinator.wait(None).await.unwrap();
    assert_eq!(result["output"], "a is done");
}

#[tokio::test]
async fn an_interrupt_persists_the_chain_state() {
    let fixture = Fixture::new(&["a"], "hop-interrupt", &HandoffConfig::default());
    let agent = FakeAgent::sync(agent_card("a"), |_inputs, _session| {
        Ok(AgentOutput::interrupt(Vec::new(), Vec::new()))
    });
    let container = fixture.container("a", &[], fake_provider(agent));
    let child = fixture.child();

    let output = container
        .invoke(request(json!({"query": "hi"}), vec![]), Some(child))
        .await
        .unwrap();
    assert_eq!(output.output, "{}");

    let result = fixture.coordinator.wait(None).await.unwrap();
    assert_eq!(result["result_type"], "interrupt");

    let mirrored = fixture
        .team_session
        .get_state(CONTEXT_HISTORY_STATE_KEY)
        .await
        .unwrap();
    assert_eq!(mirrored, None, "the hop wrote no context messages");

    let coordinator_state = fixture
        .team_session
        .get_state(COORDINATOR_STATE_KEY)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(coordinator_state["current_agent_id"], "a");
    assert_eq!(coordinator_state["handoff_count"], 0);

    let history = fixture
        .team_session
        .get_state(HISTORY_STATE_KEY)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(history.as_array().map(Vec::len), Some(1));
    assert_eq!(history[0]["agent"], "a");
    assert_eq!(history[0]["output"]["result_type"], "interrupt");
}

#[tokio::test]
async fn a_failing_target_completes_the_chain_with_its_error() {
    let fixture = Fixture::new(&["a"], "hop-error", &HandoffConfig::default());
    let agent = FakeAgent::sync(agent_card("a"), |_inputs, _session| {
        Err(Error::validation("target exploded"))
    });
    let container = fixture.container("a", &[], fake_provider(agent));
    let child = fixture.child();

    let output = container
        .invoke(request(json!({"query": "hi"}), vec![]), Some(child))
        .await
        .unwrap();
    assert_eq!(output.output, "{}");

    let error = fixture.coordinator.wait(None).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("target exploded"));
}

#[tokio::test]
async fn transfer_tools_are_injected_once_on_a_stable_target() {
    let fixture = Fixture::new(&["a", "b"], "hop-tools", &HandoffConfig::default());
    let react = Arc::new(ReActAgent::new(agent_card("a")));
    let target_provider: AgentProvider = {
        let react = Arc::clone(&react);
        Arc::new(move || TeamAgent::react(Arc::clone(&react)))
    };
    let container = fixture.container("a", &["b"], target_provider);
    fixture.runtime.register_agent(
        agent_card("b").with_description("b agent"),
        fake_provider(FakeAgent::echo("b")),
    );

    for _ in 0..2 {
        let child = fixture.child();
        container
            .invoke(request(json!({"query": "hi"}), vec![]), Some(child))
            .await
            .unwrap();
    }

    let tools: Vec<String> = react.tools().into_iter().map(|card| card.name).collect();
    assert_eq!(
        tools,
        vec!["transfer_to_b".to_owned()],
        "the transfer tool is mounted exactly once, even across two hops"
    );
    assert_eq!(container.allowed_targets().len(), 1);
    assert_eq!(container.allowed_targets()[0], "b");
    assert!(container.endpoint_id().ends_with("_a"));
}

#[tokio::test]
async fn a_target_without_a_tool_host_is_skipped() {
    let fixture = Fixture::new(&["a"], "hop-no-host", &HandoffConfig::default());
    let container = fixture.container("a", &["b"], fake_provider(FakeAgent::echo("a")));
    let child = fixture.child();

    container
        .invoke(request(json!({"query": "hi"}), vec![]), Some(child))
        .await
        .unwrap();
    assert_eq!(
        fixture.coordinator.wait(None).await.unwrap()["output"],
        "a:hi"
    );
}

#[tokio::test]
async fn history_is_shaped_into_the_target_input() {
    let fixture = Fixture::new(&["a"], "hop-history", &HandoffConfig::default());
    let seen: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&seen);
    let agent = FakeAgent::sync(agent_card("a"), move |inputs, _session| {
        sink.lock().push(inputs);
        Ok(AgentOutput::answer("done"))
    });
    let container = fixture.container("a", &[], fake_provider(agent));
    let history = vec![HandoffHistoryEntry {
        agent: "a".to_owned(),
        output: json!({"output": "first"}),
    }];

    let child = fixture.child();
    container
        .invoke(
            request(json!({"query": "hi", "trace": 7}), history.clone()),
            Some(child),
        )
        .await
        .unwrap();
    let child = fixture.child();
    container
        .invoke(request(json!("plain text"), history), Some(child))
        .await
        .unwrap();

    let inputs = seen.lock().clone();
    assert_eq!(inputs.len(), 2);
    assert_eq!(inputs[0]["query"], "hi");
    assert_eq!(inputs[0]["trace"], 7);
    assert_eq!(inputs[0]["handoff_history"][0]["agent"], "a");
    assert_eq!(inputs[0]["handoff_history"][0]["output"]["output"], "first");
    assert_eq!(inputs[1]["query"], "plain text");
    assert_eq!(inputs[1]["handoff_history"][0]["agent"], "a");
}

#[tokio::test]
async fn an_empty_history_passes_the_message_through() {
    let fixture = Fixture::new(&["a"], "hop-no-history", &HandoffConfig::default());
    let seen: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&seen);
    let agent = FakeAgent::sync(agent_card("a"), move |inputs, _session| {
        sink.lock().push(inputs);
        Ok(AgentOutput::answer("done"))
    });
    let container = fixture.container("a", &[], fake_provider(agent));
    let child = fixture.child();

    container
        .invoke(request(json!({"query": "hi"}), vec![]), Some(child))
        .await
        .unwrap();
    assert_eq!(seen.lock().clone(), vec![json!({"query": "hi"})]);
}

#[tokio::test]
async fn the_context_history_is_mirrored_from_and_to_the_hop() {
    let fixture = Fixture::new(&["a"], "hop-context", &HandoffConfig::default());
    // The team history is mirrored into the hop before it runs, and the hop's
    // cleaned messages are appended once, however often a hop repeats.
    for hop in 0..2 {
        fixture
            .team_session
            .update_state(json!({
                CONTEXT_HISTORY_STATE_KEY: [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi there"}
                ]
            }))
            .await
            .unwrap();
        let agent = FakeAgent::new(agent_card("a"), move |_inputs, session| {
            async move {
                    let session = session.expect("the hop session");
                    let state = session.get_state("context").await.unwrap();
                    if hop == 0 {
                        let messages = state
                            .as_ref()
                            .and_then(|state| state.get("default_context_id"))
                            .and_then(|context| context.get("messages"))
                            .and_then(Value::as_array)
                            .expect("the team history is injected into the hop");
                        assert_eq!(
                            messages.iter().map(|m| m["content"].clone()).collect::<Vec<_>>(),
                            vec![json!("hello"), json!("hi there")]
                        );
                    }
                    session
                        .update_state(json!({
                            "context": {
                                "default_context_id": {
                                    "messages": [
                                        {"role": "user", "content": "hello"},
                                        {"role": "assistant", "content": "tool turn", "tool_calls": [{"id": "1"}]},
                                        {"role": "tool", "content": "observation", "tool_call_id": "1"},
                                        {"role": "assistant", "content": "hi there"},
                                        {"role": "assistant", "content": "again"}
                                    ]
                                }
                            }
                        }))
                        .await
                        .unwrap();
                    Ok(AgentOutput::answer("done"))
                }
                .boxed()
        });
        let container = fixture.container("a", &[], fake_provider(agent));
        let child = fixture.child();
        container
            .invoke(request(json!({"query": "hi"}), vec![]), Some(child))
            .await
            .unwrap();
    }

    let mirrored = fixture
        .team_session
        .get_state(CONTEXT_HISTORY_STATE_KEY)
        .await
        .unwrap()
        .unwrap();
    let contents: Vec<String> = mirrored
        .as_array()
        .unwrap()
        .iter()
        .map(|message| message["content"].as_str().unwrap_or_default().to_owned())
        .collect();
    assert_eq!(
        contents,
        vec!["hello", "hi there", "again"],
        "tool turns are dropped and repeats are not appended twice"
    );
}
