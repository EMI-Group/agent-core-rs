#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Integration tests for `openjiuwen_agent_evolving::rails`.
//!
//! The suite is fully offline and deterministic: it drives the rail's
//! synchronous hooks directly with a hand-built [`HarnessRailContext`] and a
//! recording [`EvolutionObserver`], so no agent, model, or task loop runs.
//!
//! The session case uses the kernel's real [`AgentSession`], which is a
//! single-argument, in-memory session — no fake is needed. The no-session case is
//! covered separately because Python callers may invoke an agent without one.
use std::sync::{Arc, Mutex};

use openjiuwen_agent_evolving::rails::{
    registerable_rail, EvolutionEvent, EvolutionObserver, EvolutionRailSpec, HarnessEvolutionRail,
    EVOLUTION_PRIORITY,
};
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_core::session::{AgentSession, Session};
use openjiuwen_core::single_agent::RailEvent;
use openjiuwen_harness::{EvolutionRail, HarnessRail, HarnessRailContext, TaskIterationInfo};

/// Records every event it is handed so a test can assert on the payload.
struct RecordingObserver {
    events: Mutex<Vec<EvolutionEvent>>,
}

impl RecordingObserver {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            events: Mutex::new(Vec::new()),
        })
    }

    fn events(&self) -> Vec<EvolutionEvent> {
        self.events.lock().expect("observer lock").clone()
    }
}

impl EvolutionObserver for RecordingObserver {
    fn observe(&self, event: &EvolutionEvent) -> Result<()> {
        self.events
            .lock()
            .expect("observer lock")
            .push(event.clone());
        Ok(())
    }
}

/// An observer that always fails, used to check error propagation.
struct FailingObserver;

impl EvolutionObserver for FailingObserver {
    fn observe(&self, _event: &EvolutionEvent) -> Result<()> {
        Err(Error::unsupported("boom"))
    }
}

/// Builds a context with an agent id, a loop iteration, and a real session.
fn context_with_session(event: RailEvent) -> HarnessRailContext {
    let mut ctx = HarnessRailContext::for_event(event);
    "agent-1".clone_into(&mut ctx.base.agent_id);
    ctx.base.iteration = 2;
    let session: Arc<dyn Session> = Arc::new(AgentSession::new("sess-1"));
    ctx.base.session = Some(session);
    ctx
}

/// Builds a context with an agent id and a loop iteration but no session.
fn context_without_session(event: RailEvent) -> HarnessRailContext {
    let mut ctx = HarnessRailContext::for_event(event);
    "agent-1".clone_into(&mut ctx.base.agent_id);
    ctx.base.iteration = 2;
    ctx
}

#[test]
fn priority_matches_the_python_rail_ladder() {
    assert_eq!(EVOLUTION_PRIORITY, 60);
    assert_eq!(
        EVOLUTION_PRIORITY,
        openjiuwen_harness::rail_priority::EVOLUTION
    );
}

#[test]
fn before_invoke_records_the_invocation() {
    let sink = RecordingObserver::new();
    let rail = HarnessEvolutionRail::new(sink.clone());
    let mut ctx = context_with_session(RailEvent::BeforeInvoke);

    rail.on_before_invoke(&mut ctx).expect("observe");

    assert_eq!(
        sink.events(),
        [EvolutionEvent::InvokeStarted {
            agent_id: "agent-1".to_owned(),
            session_id: Some("sess-1".to_owned()),
            iteration: 2,
        }]
    );
}

#[test]
fn before_invoke_reports_no_session_as_none() {
    let sink = RecordingObserver::new();
    let rail = HarnessEvolutionRail::new(sink.clone());
    let mut ctx = context_without_session(RailEvent::BeforeInvoke);

    rail.on_before_invoke(&mut ctx).expect("observe");

    assert_eq!(
        sink.events(),
        [EvolutionEvent::InvokeStarted {
            agent_id: "agent-1".to_owned(),
            session_id: None,
            iteration: 2,
        }]
    );
}

#[test]
fn event_json_uses_the_snake_case_type_tag() {
    let event = EvolutionEvent::InvokeStarted {
        agent_id: "agent-1".to_owned(),
        session_id: Some("sess-1".to_owned()),
        iteration: 2,
    };
    let json = serde_json::to_value(&event).expect("serialize");
    assert_eq!(
        json,
        serde_json::json!({
            "type": "invoke_started",
            "agent_id": "agent-1",
            "session_id": "sess-1",
            "iteration": 2,
        })
    );

    let without_session = EvolutionEvent::InvokeStarted {
        agent_id: "agent-1".to_owned(),
        session_id: None,
        iteration: 2,
    };
    let json = serde_json::to_value(&without_session).expect("serialize");
    assert_eq!(json["session_id"], serde_json::Value::Null);
}

#[test]
fn task_iteration_uses_the_task_info() {
    let sink = RecordingObserver::new();
    let rail = HarnessEvolutionRail::new(sink.clone());
    let mut ctx =
        context_with_session(RailEvent::AfterInvoke).with_task(TaskIterationInfo::new(7, 3));

    rail.on_after_task_iteration(&mut ctx).expect("observe");

    assert_eq!(
        sink.events(),
        [EvolutionEvent::TaskIterationCompleted {
            agent_id: "agent-1".to_owned(),
            session_id: Some("sess-1".to_owned()),
            round_id: 7,
            iteration: 3,
            is_follow_up: false,
        }]
    );
}

#[test]
fn task_iteration_marks_a_follow_up() {
    let sink = RecordingObserver::new();
    let rail = HarnessEvolutionRail::new(sink.clone());
    let mut ctx = context_with_session(RailEvent::AfterInvoke)
        .with_task(TaskIterationInfo::new(7, 3).with_follow_up());

    rail.on_after_task_iteration(&mut ctx).expect("observe");

    let events = sink.events();
    assert_eq!(events.len(), 1);
    match &events[0] {
        EvolutionEvent::TaskIterationCompleted {
            round_id,
            iteration,
            is_follow_up,
            ..
        } => {
            assert_eq!(*round_id, 7);
            assert_eq!(*iteration, 3);
            assert!(is_follow_up);
        }
        other @ EvolutionEvent::InvokeStarted { .. } => {
            panic!("unexpected event: {other:?}")
        }
    }
}

#[test]
fn task_iteration_without_info_falls_back_to_the_base_iteration() {
    let sink = RecordingObserver::new();
    let rail = HarnessEvolutionRail::new(sink.clone());
    let mut ctx = context_with_session(RailEvent::AfterInvoke);

    rail.on_after_task_iteration(&mut ctx).expect("observe");

    assert_eq!(
        sink.events(),
        [EvolutionEvent::TaskIterationCompleted {
            agent_id: "agent-1".to_owned(),
            session_id: Some("sess-1".to_owned()),
            round_id: 0,
            iteration: 2,
            is_follow_up: false,
        }]
    );
}

#[test]
fn task_iteration_json_uses_the_snake_case_type_tag() {
    let event = EvolutionEvent::TaskIterationCompleted {
        agent_id: "agent-1".to_owned(),
        session_id: None,
        round_id: 7,
        iteration: 3,
        is_follow_up: true,
    };
    let json = serde_json::to_value(&event).expect("serialize");
    assert_eq!(
        json,
        serde_json::json!({
            "type": "task_iteration_completed",
            "agent_id": "agent-1",
            "session_id": null,
            "round_id": 7,
            "iteration": 3,
            "is_follow_up": true,
        })
    );

    let decoded: EvolutionEvent = serde_json::from_value(json).expect("deserialize");
    assert_eq!(decoded, event);
}

#[test]
fn observer_errors_propagate_unchanged() {
    let rail = HarnessEvolutionRail::new(Arc::new(FailingObserver));

    let mut before = context_with_session(RailEvent::BeforeInvoke);
    let error = rail.on_before_invoke(&mut before).expect_err("must fail");
    assert!(matches!(error, Error::Unsupported(_)));
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert_eq!(error.message(), "boom");

    let mut after = context_with_session(RailEvent::AfterInvoke);
    let error = rail
        .on_after_task_iteration(&mut after)
        .expect_err("must fail");
    assert!(matches!(error, Error::Unsupported(_)));
    assert_eq!(error.message(), "boom");
}

#[test]
fn default_spec_is_evolution_60_enabled() {
    let rail = HarnessEvolutionRail::new(RecordingObserver::new());
    let spec = rail.spec();
    assert_eq!(spec.name(), "evolution");
    assert_eq!(spec.priority(), EVOLUTION_PRIORITY);
    assert!(spec.enabled());
    assert_eq!(spec, &EvolutionRailSpec::default());
    assert_eq!(rail.name(), "evolution");
    assert_eq!(rail.priority(), 60);
}

#[test]
fn custom_spec_builders_are_honoured() {
    let sink = RecordingObserver::new();
    let spec = EvolutionRailSpec::default()
        .with_name("custom-evolution")
        .with_priority(15)
        .with_enabled(false);
    let rail = HarnessEvolutionRail::new(sink.clone()).with_spec(spec.clone());

    assert_eq!(rail.spec(), &spec);
    assert_eq!(rail.name(), "custom-evolution");
    assert_eq!(rail.priority(), 15);
    assert!(!rail.spec().enabled());
    // The observer is still attached after replacing the spec.
    let mut ctx = context_with_session(RailEvent::BeforeInvoke);
    rail.on_before_invoke(&mut ctx).expect("observe");
    assert_eq!(sink.events().len(), 1);
}

#[test]
fn spec_round_trips_through_json_and_tolerates_partial_input() {
    let spec = EvolutionRailSpec::default();
    let json = serde_json::to_value(&spec).expect("serialize");
    assert_eq!(
        json,
        serde_json::json!({"name": "evolution", "priority": 60, "enabled": true})
    );
    let decoded: EvolutionRailSpec = serde_json::from_value(json).expect("deserialize");
    assert_eq!(decoded, spec);

    let partial: EvolutionRailSpec =
        serde_json::from_str(r#"{"name":"custom","future":1}"#).expect("deserialize");
    assert_eq!(partial.name(), "custom");
    assert_eq!(partial.priority(), EVOLUTION_PRIORITY);
    assert!(partial.enabled());
}

#[test]
fn registerable_rail_exposes_the_spec_to_the_harness() {
    let rail: Arc<dyn HarnessRail> =
        registerable_rail(HarnessEvolutionRail::new(RecordingObserver::new()));
    assert_eq!(rail.name(), "evolution");
    assert_eq!(rail.priority(), 60);
}

#[test]
fn registerable_rail_forwards_a_custom_spec() {
    let spec = EvolutionRailSpec::default()
        .with_name("custom-evolution")
        .with_priority(15);
    let rail: Arc<dyn HarnessRail> =
        registerable_rail(HarnessEvolutionRail::new(RecordingObserver::new()).with_spec(spec));
    assert_eq!(rail.name(), "custom-evolution");
    assert_eq!(rail.priority(), 15);
}

#[tokio::test]
async fn adapter_swallows_observer_errors() {
    // The harness adapter logs and swallows evolution failures, so a broken
    // observer never aborts a run.
    let rail = registerable_rail(HarnessEvolutionRail::new(Arc::new(FailingObserver)));
    let mut ctx = context_with_session(RailEvent::BeforeInvoke);
    rail.before_invoke(&mut ctx)
        .await
        .expect("adapter swallows");
}
