//! Human-in-the-loop hub behaviour: the live-run registry, inbox delivery,
//! cancellation, the typed pause/resume refusals, real interaction round trips
//! through a session, and the run-path seam that registers and finalises a run.
//!
//! Drives only the public `openjiuwen_runner::hitl` surface plus the
//! `run::RunServices::with_monitor` / `with_hitl` seams, exactly as a host UI and
//! the execution paths do. Ports `team_runner.py`'s `interact_agent_team`
//! (delivery + `DeliverResult`), `register_human_agent_inbound` (the inbound
//! channel), `pause_agent_team` (kept as a typed refusal: nothing suspends an
//! in-flight invocation), and `stop_agent_team` (a real cancellation).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use openjiuwen_agent::Agent;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_runner::hitl::{
    ActiveRunSpec, DeliverResult, InteractKind, InteractPayload, InteractionHub, RunTarget,
    INBOX_CAPACITY,
};
use openjiuwen_runner::monitor::{RunKind, RunMonitor, RunPhase};
use openjiuwen_runner::run::{invoke_agent, stream_agent, AgentRef, RunOptions};
use openjiuwen_session::{
    AgentSession, InteractionDecision, InteractionRequest, InteractionResponse, Session,
};
use serde_json::json;
use tokio_util::sync::CancellationToken;

use common::{answer_frames, scripted_agent, ScriptedAgent};

/// Upper bound on a deterministic in-memory round trip.
const BOUND: Duration = Duration::from_secs(10);

/// A spec labelled `assistant` bound to `session_id`, with no team.
fn session_spec(session_id: &str) -> ActiveRunSpec {
    ActiveRunSpec {
        session_id: Some(session_id.to_string()),
        ..ActiveRunSpec::new("assistant")
    }
}

/// The phases a run's retained history records, oldest first.
fn phases(monitor: &RunMonitor, run_id: &str) -> Vec<RunPhase> {
    monitor
        .history(run_id)
        .expect("a tracked run")
        .into_iter()
        .map(|event| event.phase)
        .collect()
}

/// Polls `hub.pending(run_id)` until it is non-empty, bounded by [`BOUND`].
async fn wait_pending(hub: &InteractionHub, run_id: &str) {
    tokio::time::timeout(BOUND, async {
        loop {
            if !hub.pending(run_id).is_empty() {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the interaction frame reaches the hub watcher before the deadline");
}

/// Registers a run and attaches a fresh `AgentSession` named `session_id`.
fn attached_run(hub: &InteractionHub, session_id: &str) -> (String, Arc<AgentSession>) {
    let handle = hub.register(session_spec(session_id));
    let run_id = handle.run_id().to_string();
    let session = Arc::new(AgentSession::new(session_id));
    hub.attach_session(&run_id, Arc::clone(&session) as Arc<dyn Session>)
        .expect("the run is registered");
    (run_id, session)
}

// ---------------------------------------------------------------------------
// Registration and delivery
// ---------------------------------------------------------------------------

// ports team_runner.py::interact_agent_team (delivery + DeliverResult)
#[tokio::test]
async fn register_deliver_and_take_inbox_round_trip() {
    let hub = InteractionHub::new();
    let handle = hub.register(ActiveRunSpec {
        team_name: Some("crew".to_string()),
        ..session_spec("s-1")
    });
    let run_id = handle.run_id().to_string();

    let payload = InteractPayload::body(InteractKind::Operator, json!({"instruction": "go"}))
        .with_sender("operator-7")
        .with_text("go");
    let result = hub.deliver(RunTarget::for_team("crew"), payload.clone());
    assert_eq!(result, DeliverResult::Accepted);
    assert!(result.success());

    let mut inbox = hub.take_inbox(&run_id).expect("the run's inbox");
    let received = tokio::time::timeout(BOUND, inbox.recv())
        .await
        .expect("the payload is queued")
        .expect("the payload is delivered");
    assert_eq!(received, payload, "the payload is forwarded verbatim");
    assert!(
        hub.take_inbox(&run_id).is_none(),
        "the inbox is handed out exactly once"
    );
    assert!(inbox.try_recv().is_err(), "and it carries one payload");
}

#[test]
fn an_unmatched_target_reports_missing() {
    let hub = InteractionHub::new();
    let _team = hub.register(ActiveRunSpec {
        team_name: Some("crew".to_string()),
        ..ActiveRunSpec::new("crew")
    });
    let _bare = hub.register(ActiveRunSpec::new("lonely"));

    for result in [
        hub.deliver(RunTarget::for_team("other"), InteractPayload::text("hi")),
        hub.deliver(RunTarget::new(), InteractPayload::text("hi")),
        hub.deliver(
            RunTarget::for_session("s-unused"),
            InteractPayload::text("hi"),
        ),
    ] {
        assert_eq!(result, DeliverResult::MissingTarget);
        assert!(!result.success());
    }
}

#[tokio::test]
async fn a_closed_or_full_inbox_is_reported_never_dropped_silently() {
    let hub = InteractionHub::new();

    let closed = hub.register(session_spec("s-closed"));
    let inbox = hub.take_inbox(closed.run_id()).expect("the run's inbox");
    drop(inbox);
    let target = RunTarget::for_session("s-closed");
    let result = hub.deliver(target.clone(), InteractPayload::text("hi"));
    assert_eq!(
        result,
        DeliverResult::Closed,
        "a dropped receiver is a closed inbox"
    );
    assert!(!result.success());
    assert_eq!(
        hub.deliver(target, InteractPayload::text("again")),
        DeliverResult::NoInbox,
        "the dead sender is dropped after the first failure"
    );

    let full = hub.register(session_spec("s-full"));
    let _inbox = hub
        .take_inbox(full.run_id())
        .expect("the run's inbox is never drained");
    let target = RunTarget::for_session("s-full");
    for _ in 0..INBOX_CAPACITY {
        assert_eq!(
            hub.deliver(target.clone(), InteractPayload::text("x")),
            DeliverResult::Accepted
        );
    }
    let overflow = hub.deliver(target, InteractPayload::text("overflow"));
    assert_eq!(
        overflow,
        DeliverResult::Closed,
        "a full inbox is reported, not silently dropped"
    );
    assert!(!overflow.success());
}

// ---------------------------------------------------------------------------
// Pause/resume refusals and cancellation
// ---------------------------------------------------------------------------

// ports team_runner.py::pause_agent_team (a typed refusal) / stop_agent_team
#[test]
fn pause_and_resume_are_refused_without_changing_the_run() {
    let monitor = Arc::new(RunMonitor::new());
    let hub = InteractionHub::new().with_monitor(Arc::clone(&monitor));
    let handle = hub.register(ActiveRunSpec {
        team_name: Some("crew".to_string()),
        ..session_spec("s-1")
    });
    let run_id = handle.run_id().to_string();

    for (action, result) in [
        ("pause", hub.pause(&run_id)),
        ("resume", hub.resume(&run_id)),
    ] {
        let error = result.expect_err("a live run cannot be suspended");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        let message = error.to_string();
        assert!(
            message.contains("InteractGate") && message.contains("cancel"),
            "the refusal names the missing seam and the alternative: {message}"
        );
        assert!(
            message.contains(&format!("`{action}`")) && message.contains(&run_id),
            "the refusal names the call and the run: {message}"
        );
    }

    let info = hub.list_active().remove(0);
    assert_eq!(info.state, RunPhase::Queued, "the refusal changed no phase");
    assert!(!info.gate_closed, "the refusal left the gate open");
    assert_eq!(
        phases(&monitor, &run_id),
        vec![RunPhase::Queued],
        "a refused pause/resume publishes no monitor event"
    );

    // Cancelling still works end to end: terminal phase, closed gate, one event.
    hub.cancel(&run_id).expect("cancel");
    let info = hub.list_active().remove(0);
    assert_eq!(info.state, RunPhase::Cancelled);
    assert!(info.gate_closed);
    assert_eq!(
        phases(&monitor, &run_id),
        vec![RunPhase::Queued, RunPhase::Cancelled]
    );
}

#[test]
fn a_cancelled_run_rejects_pause_and_resume_as_validation() {
    let hub = InteractionHub::new();
    let handle = hub.register(session_spec("s-1"));
    let run_id = handle.run_id().to_string();
    hub.cancel(&run_id).expect("cancel");

    for result in [hub.pause(&run_id), hub.resume(&run_id)] {
        let error = result.expect_err("a cancelled run cannot be paused or resumed");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(
            error.to_string(),
            format!("validation: run '{run_id}' has already been cancelled")
        );
    }
}

#[test]
fn an_unknown_run_is_not_found_on_pause_and_resume() {
    let hub = InteractionHub::new();
    for result in [hub.pause("run-x"), hub.resume("run-x")] {
        let error = result.expect_err("an unknown run");
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert_eq!(
            error.to_string(),
            "not_found: run 'run-x' is not registered"
        );
    }
}

#[tokio::test]
async fn cancel_trips_the_runs_own_token_and_is_observable() {
    let hub = InteractionHub::new();
    let token = CancellationToken::new();
    let handle = hub.register(ActiveRunSpec {
        cancellation: Some(token.clone()),
        ..session_spec("s-1")
    });

    hub.cancel(handle.run_id()).expect("cancel");
    assert!(token.is_cancelled(), "the run's own token was tripped");
    assert!(handle.cancellation().is_cancelled());

    // The run path selects on the token; prove the wait resolves rather than only
    // reading a flag a future would keep ignoring.
    tokio::time::timeout(BOUND, async {
        tokio::select! {
            biased;
            () = token.cancelled() => {}
            () = std::future::pending::<()>() => panic!("the token never fired"),
        }
    })
    .await
    .expect("the cancellation is observable before the deadline");
}

// ---------------------------------------------------------------------------
// Interactions through a session
// ---------------------------------------------------------------------------

#[tokio::test]
async fn an_interaction_round_trip_resolves_through_the_hub() {
    let hub = InteractionHub::new();
    let (run_id, session) = attached_run(&hub, "s-1");

    let task = {
        let session = Arc::clone(&session);
        tokio::spawn(async move {
            session
                .request_interaction(InteractionRequest::ask_user(json!({"prompt": "ok?"})))
                .await
        })
    };

    wait_pending(&hub, &run_id).await;
    let pending = hub.pending(&run_id);
    assert_eq!(pending.len(), 1, "the hub observed exactly one request");
    let request_id = pending[0].clone();
    assert_eq!(hub.list_active().remove(0).queued, 1);

    hub.answer(InteractionResponse::answer(
        request_id.clone(),
        json!("yes"),
    ))
    .await
    .expect("the request is pending");

    let response = tokio::time::timeout(BOUND, task)
        .await
        .expect("the request resolves before the deadline")
        .expect("the request task does not panic")
        .expect("the answer is accepted");
    assert_eq!(response.request_id, request_id);
    assert_eq!(response.decision, InteractionDecision::Answer);
    assert_eq!(response.value, json!("yes"));
    assert_eq!(
        hub.pending(&run_id),
        Vec::<String>::new(),
        "the id was resolved"
    );
    assert_eq!(hub.list_active().remove(0).queued, 0);
}

#[tokio::test]
async fn an_unknown_interaction_id_is_not_found() {
    let hub = InteractionHub::new();
    let (run_id, _session) = attached_run(&hub, "s-1");

    let error = hub
        .answer(InteractionResponse::answer("intr_missing", json!(1)))
        .await
        .expect_err("no run accepts an unknown interaction id");
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(
        error.to_string().contains("intr_missing"),
        "the message names the interaction: {error}"
    );
    assert_eq!(hub.pending(&run_id), Vec::<String>::new());
}

#[tokio::test]
async fn close_interactions_cancels_a_waiting_request() {
    let hub = InteractionHub::new();
    let (run_id, session) = attached_run(&hub, "s-1");

    // No per-request timeout: the waiter blocks until the hub resolves it.
    let request = InteractionRequest::ask_user(json!({"prompt": "again?"}));
    let request_id = request.id.clone();
    let task = {
        let session = Arc::clone(&session);
        tokio::spawn(async move { session.request_interaction(request).await })
    };

    wait_pending(&hub, &run_id).await;
    hub.close_interactions("shutdown")
        .await
        .expect("the hub-wide close succeeds");
    assert_eq!(hub.pending(&run_id), Vec::<String>::new());

    let response = tokio::time::timeout(BOUND, task)
        .await
        .expect("the request resolves before the deadline")
        .expect("the request task does not panic")
        .expect("the cancellation is delivered");
    assert_eq!(response.request_id, request_id);
    assert_eq!(response.decision, InteractionDecision::Cancel);
    assert_eq!(response.reason.as_deref(), Some("shutdown"));
}

// ---------------------------------------------------------------------------
// The run-path seam
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_agent_run_path_mirrors_into_the_monitor_and_the_hub() {
    // A completed run is admitted, marked running, and finished once.
    let monitor = Arc::new(RunMonitor::new());
    let services = common::services().with_monitor(Arc::clone(&monitor));
    let mut events = monitor.subscribe();
    let agent = scripted_agent("assistant", "hi back");

    let output = invoke_agent(
        services,
        AgentRef::of(Arc::clone(&agent) as Arc<dyn Agent>),
        json!({"query": "hi"}),
        RunOptions::default(),
    )
    .await
    .expect("the scripted run succeeds");
    assert_eq!(output.output, "hi back");

    let published: Vec<_> = std::iter::from_fn(|| events.try_recv().ok()).collect();
    assert_eq!(
        published
            .iter()
            .map(|event| event.phase)
            .collect::<Vec<_>>(),
        vec![RunPhase::Queued, RunPhase::Running, RunPhase::Completed],
        "one event per transition"
    );
    let run_id = published[0].run_id.clone();
    assert!(
        published.iter().all(|event| event.run_id == run_id),
        "exactly one run is tracked per invocation"
    );

    let status = monitor
        .snapshot(&run_id)
        .expect("the finished run is retained");
    assert_eq!(status.kind, RunKind::Agent);
    assert_eq!(status.label, "assistant");
    assert_eq!(status.phase, RunPhase::Completed);
    assert!(status.finished_at.is_some());
    assert!(status.error.is_none());
    assert_eq!(monitor.list_active(), Vec::new(), "the run finished");
    let session_id = status.session_id.expect("the run reports its session");
    assert_eq!(
        agent.session_ids(),
        vec![session_id],
        "the reported session is the one the agent ran in"
    );

    // A pre-cancelled run finishes cancelled.
    let cancelled = Arc::new(RunMonitor::new());
    let services = common::services().with_monitor(Arc::clone(&cancelled));
    let mut events = cancelled.subscribe();
    let token = CancellationToken::new();
    token.cancel();
    let error = invoke_agent(
        services,
        AgentRef::of(scripted_agent("assistant", "never") as Arc<dyn Agent>),
        json!({}),
        RunOptions::default().with_cancellation(token),
    )
    .await
    .expect_err("a pre-cancelled run reports cancellation");
    assert_eq!(error.kind(), ErrorKind::Cancelled);

    let published: Vec<_> = std::iter::from_fn(|| events.try_recv().ok()).collect();
    let run_id = published[0].run_id.clone();
    let status = cancelled.snapshot(&run_id).expect("the run is retained");
    assert_eq!(status.phase, RunPhase::Cancelled);
    assert!(status.finished_at.is_some());
    assert_eq!(cancelled.list_active(), Vec::new());
}

#[tokio::test]
async fn the_hub_registers_the_run_and_removes_it_again() {
    let hub = Arc::new(InteractionHub::new());
    let services = common::services().with_hitl(Arc::clone(&hub));
    let started = Arc::new(tokio::sync::Notify::new());
    let gate = Arc::new(tokio::sync::Notify::new());
    let agent = Arc::new(
        ScriptedAgent::new("assistant")
            .notifying_start(Arc::clone(&started))
            .blocking_on(Arc::clone(&gate)),
    );

    let running = tokio::spawn(invoke_agent(
        services,
        AgentRef::of(Arc::clone(&agent) as Arc<dyn Agent>),
        json!({}),
        RunOptions::default().with_session_id("hitl-run"),
    ));

    started.notified().await;
    let active = hub.list_active();
    assert_eq!(active.len(), 1, "the live run is registered");
    assert_eq!(active[0].label, "assistant");
    assert_eq!(active[0].session_id.as_deref(), Some("hitl-run"));
    assert_eq!(active[0].state, RunPhase::Queued);
    assert_eq!(active[0].queued, 0);

    gate.notify_one();
    running
        .await
        .expect("the run task does not panic")
        .expect("the run succeeds");

    assert_eq!(hub.list_active(), Vec::new(), "the run was unregistered");
    assert_eq!(hub.len(), 0);
}

#[tokio::test]
async fn dropping_a_run_stream_cancels_its_mirrored_run() {
    let monitor = Arc::new(RunMonitor::new());
    let services = common::services().with_monitor(Arc::clone(&monitor));
    let mut events = monitor.subscribe();
    let agent = Arc::new(ScriptedAgent::new("assistant").with_chunks(answer_frames()));

    let mut stream = stream_agent(
        services,
        AgentRef::of(Arc::clone(&agent) as Arc<dyn Agent>),
        json!({"query": "hi"}),
        RunOptions::default(),
    );
    let first = stream
        .next()
        .await
        .expect("a first frame")
        .expect("a frame");
    assert_eq!(first.index, 0);

    let admitted = events
        .try_recv()
        .expect("the run is admitted before its first frame");
    assert_eq!(admitted.phase, RunPhase::Queued);
    let run_id = admitted.run_id;
    assert_eq!(monitor.list_active().len(), 1, "the live run is listed");

    drop(stream);

    tokio::time::timeout(BOUND, async {
        loop {
            if monitor.list_active().is_empty() {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("a dropped stream leaves no live run behind");

    let status = monitor
        .snapshot(&run_id)
        .expect("the finished run is retained");
    assert_eq!(status.phase, RunPhase::Cancelled);
    assert_eq!(status.detail, json!({"reason": "run dropped"}));
    assert!(status.finished_at.is_some());
    assert!(status.error.is_none());
}
