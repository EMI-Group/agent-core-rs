//! Runner-level human-in-the-loop (HITL) hub for active runs.
//!
//! An [`InteractionHub`] is the broker between a host UI and the runs a runner is
//! executing. It keeps the live-run registry ([`ActiveRunInfo`]), delivers inbound
//! human payloads to a run's bounded inbox ([`InteractionHub::deliver`]), answers
//! the interaction requests a run's session published ([`InteractionHub::answer`]),
//! and exposes pause/resume/cancel as coordination signals
//! ([`InteractionHub::pause`], [`InteractionHub::resume`],
//! [`InteractionHub::cancel`]).
//!
//! **Python correspondence.** The hub is the Rust form of `team_runner.py`'s
//! `interact_agent_team` (deliver a payload to an active team runtime and report
//! a `DeliverResult`), `register_human_agent_inbound` (the team→user channel the
//! [`InteractKind`] variants model: god-view, operator, human-agent, group chat),
//! and `pause_agent_team` / `stop_agent_team`. [`InteractPayload`] is the union of
//! Python's `GodViewMessage` / `OperatorMessage` / `HumanAgentMessage` /
//! `GroupChatMessage`, plus [`InteractKind::Text`] for the bare-string shorthand.
//!
//! **Signals, not a scheduler.** [`InteractionHub::pause`] and
//! [`InteractionHub::resume`] flip the run's hub state and close/open its gate so a
//! host can coordinate and render; they do not suspend the agent or team loop.
//! In-loop gate enforcement is Python's `InteractGate`, which is deferred (see
//! `CONTEXT.md`).
//!
//! **In-process only.** The hub lives in the runner process: there is no message
//! queue, websocket, or Pulsar transport between processes, no reply-topic
//! routing, and no authentication of who may answer an interaction. Its state is
//! in memory and lost when the hub is dropped.
//!
//! **Best-effort monitor mirroring.** When a monitor is attached
//! ([`InteractionHub::with_monitor`]) the hub mirrors its transitions into it;
//! the hub's own state is authoritative and a rejected mirror transition is
//! logged, never returned.

mod bridge;
mod hub;
mod payload;
mod target;

pub use bridge::{ActiveRunInfo, ActiveRunSpec, RunHandle};
pub use hub::{InteractionHub, INBOX_CAPACITY};
pub use payload::{InteractKind, InteractPayload};
pub use target::{DeliverResult, RunTarget};

// The execution paths adopt this bridge (`bind_run`/`RunBinding`); until they do,
// the seam is intentionally unreferenced inside the crate.
#[allow(unused_imports)]
pub(crate) use bridge::{bind_run, RunBinding};

use openjiuwen_common::error::Error;

/// The tracing target every hub log line uses.
pub(crate) const TARGET: &str = "openjiuwen_runner::hitl";

/// The reason recorded on the cancellation responses the hub sends.
pub(crate) const CANCEL_REASON: &str = "run cancelled";

// ------------------------------------------------------- hub internals --

/// `run '<id>' is not registered`: the id was never registered or was removed.
fn not_registered(run_id: &str) -> Error {
    crate::error::not_found("run", run_id)
}

/// `run '<id>' has already been cancelled`: the hub's state machine rejects the
/// transition.
fn already_cancelled(run_id: &str) -> Error {
    Error::Validation(format!("run '{run_id}' has already been cancelled"))
}

/// `run '<id>' has already finished`: reached only when a caller assigned a
/// terminal phase to a registered run (the hub itself removes finished runs).
fn already_finished(run_id: &str) -> Error {
    Error::Validation(format!("run '{run_id}' has already finished"))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

    use std::sync::Arc;
    use std::time::Duration;

    use openjiuwen_common::error::{ErrorKind, Result};
    use openjiuwen_session::interaction::{
        InteractionDecision, InteractionRequest, InteractionResponse,
    };
    use openjiuwen_session::{AgentSession, Session};
    use serde_json::json;
    use tokio_util::sync::CancellationToken;

    use crate::monitor::{RunKind, RunMonitor, RunOutcome, RunPhase};

    use super::*;

    /// Upper bound on a deterministic in-memory round trip.
    const WAIT: Duration = Duration::from_secs(10);

    fn spec(label: &str, session: Option<&str>) -> ActiveRunSpec {
        ActiveRunSpec {
            session_id: session.map(str::to_string),
            ..ActiveRunSpec::new(label)
        }
    }

    fn payload(text: &str) -> InteractPayload {
        InteractPayload::text(text)
    }

    /// Polls `hub.pending` until `expected` ids are observed, bounded by [`WAIT`].
    async fn wait_pending(hub: &InteractionHub, run_id: &str, expected: usize) {
        let wait = async {
            loop {
                if hub.pending(run_id).len() >= expected {
                    return;
                }
                tokio::task::yield_now().await;
            }
        };
        tokio::time::timeout(WAIT, wait)
            .await
            .expect("the interaction frame must reach the hub watcher before the deadline");
    }

    fn assert_not_registered<T: std::fmt::Debug>(result: Result<T>) {
        let error = result.expect_err("unknown run id");
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert_eq!(
            error.to_string(),
            "not_found: run 'run-x' is not registered"
        );
    }

    #[test]
    fn register_mints_ids_unless_supplied() {
        let hub = InteractionHub::new();
        let first = hub.register(ActiveRunSpec::new("assistant"));
        let second = hub.register(ActiveRunSpec::new("assistant"));
        assert_eq!(first.run_id().len(), 36, "a hyphenated UUID v4");
        assert_ne!(first.run_id(), second.run_id());
        assert_eq!(hub.len(), 2);
        assert!(!hub.is_empty());

        let explicit = hub.register(ActiveRunSpec {
            run_id: Some("run-1".to_string()),
            ..ActiveRunSpec::new("assistant")
        });
        assert_eq!(explicit.run_id(), "run-1");
        assert_eq!(hub.len(), 3);
    }

    #[test]
    fn register_replaces_an_existing_run_id() {
        let hub = InteractionHub::new();
        let first_spec = ActiveRunSpec {
            run_id: Some("run-1".to_string()),
            ..spec("first", Some("s-1"))
        };
        let _first = hub.register(first_spec);
        let info = hub.list_active().remove(0);
        assert_eq!(info.label, "first");

        let _second = hub.register(ActiveRunSpec {
            run_id: Some("run-1".to_string()),
            ..spec("second", Some("s-2"))
        });
        assert_eq!(hub.len(), 1);
        let info = hub.list_active().remove(0);
        assert_eq!(info.label, "second");
        assert_eq!(info.session_id.as_deref(), Some("s-2"));
        assert!(
            hub.take_inbox("run-1").is_some(),
            "the replacement inbox is live"
        );
    }

    #[tokio::test]
    async fn delivery_finds_the_matching_run_and_take_inbox_receives_once() {
        let hub = InteractionHub::new();
        let handle = hub.register(ActiveRunSpec {
            team_name: Some("crew".to_string()),
            ..spec("assistant", Some("s-1"))
        });
        let mut inbox = hub.take_inbox(handle.run_id()).expect("a fresh inbox");

        let result = hub.deliver(RunTarget::for_session("s-1"), payload("hi"));
        assert_eq!(result, DeliverResult::Accepted);
        assert!(result.success());
        let received = inbox.recv().await.expect("the payload");
        assert_eq!(received.text.as_deref(), Some("hi"));
        assert!(hub.take_inbox(handle.run_id()).is_none(), "handed out once");

        let target = RunTarget {
            team_name: Some("crew".to_string()),
            session_id: Some("s-1".to_string()),
        };
        assert_eq!(
            hub.deliver(target, payload("again")),
            DeliverResult::Accepted
        );
        assert_eq!(
            inbox.recv().await.expect("the payload").text.as_deref(),
            Some("again")
        );
    }

    #[test]
    fn an_unmatched_target_is_missing() {
        let hub = InteractionHub::new();
        let _handle = hub.register(ActiveRunSpec {
            team_name: Some("crew".to_string()),
            ..spec("assistant", Some("s-1"))
        });
        assert_eq!(
            hub.deliver(RunTarget::new(), payload("hi")),
            DeliverResult::MissingTarget
        );
        assert_eq!(
            hub.deliver(RunTarget::for_team("other"), payload("hi")),
            DeliverResult::MissingTarget
        );
        assert_eq!(
            hub.deliver(RunTarget::for_session("s-2"), payload("hi")),
            DeliverResult::MissingTarget
        );
        assert!(InteractionHub::new().is_empty());
    }

    #[test]
    fn a_dropped_receiver_is_closed_then_no_inbox() {
        let hub = InteractionHub::new();
        let handle = hub.register(spec("assistant", Some("s-1")));
        let inbox = hub.take_inbox(handle.run_id()).expect("a fresh inbox");
        drop(inbox);

        let target = RunTarget::for_session("s-1");
        assert_eq!(
            hub.deliver(target.clone(), payload("hi")),
            DeliverResult::Closed
        );
        assert_eq!(hub.deliver(target, payload("hi")), DeliverResult::NoInbox);
    }

    #[test]
    fn a_full_inbox_is_reported_as_closed() {
        let hub = InteractionHub::new();
        let handle = hub.register(spec("assistant", Some("s-1")));
        let _inbox = hub.take_inbox(handle.run_id()).expect("a fresh inbox");
        let target = RunTarget::for_session("s-1");
        for _ in 0..INBOX_CAPACITY {
            assert_eq!(
                hub.deliver(target.clone(), payload("x")),
                DeliverResult::Accepted
            );
        }
        assert_eq!(
            hub.deliver(target, payload("overflow")),
            DeliverResult::Closed
        );
    }

    #[test]
    fn unknown_ids_are_not_registered() {
        let hub = InteractionHub::new();
        assert_not_registered(hub.unregister("run-x"));
        assert_not_registered(hub.attach_session("run-x", Arc::new(AgentSession::new("s-1"))));
        assert_not_registered(hub.pause("run-x"));
        assert_not_registered(hub.resume("run-x"));
        assert_not_registered(hub.cancel("run-x"));
        assert!(hub.pending("run-x").is_empty());
    }

    #[test]
    fn pause_and_resume_flip_the_state_and_gate() {
        let hub = InteractionHub::new();
        let handle = hub.register(spec("assistant", Some("s-1")));
        let run_id = handle.run_id();

        hub.pause(run_id).unwrap();
        let info = hub.list_active().remove(0);
        assert_eq!(info.state, RunPhase::Paused);
        assert!(info.gate_closed);
        hub.pause(run_id).unwrap();
        assert_eq!(hub.list_active()[0].state, RunPhase::Paused);

        hub.resume(run_id).unwrap();
        let info = hub.list_active().remove(0);
        assert_eq!(info.state, RunPhase::Running);
        assert!(!info.gate_closed);
        hub.resume(run_id).unwrap();
        assert_eq!(hub.list_active()[0].state, RunPhase::Running);

        let queued = hub.register(spec("other", None));
        hub.resume(queued.run_id()).unwrap();
        assert_eq!(hub.list_active()[1].state, RunPhase::Running);
        assert!(!hub.list_active()[1].gate_closed);
    }

    #[test]
    fn a_cancelled_run_rejects_pause_and_resume() {
        let hub = InteractionHub::new();
        let handle = hub.register(spec("assistant", Some("s-1")));
        hub.cancel(handle.run_id()).unwrap();

        for result in [hub.pause(handle.run_id()), hub.resume(handle.run_id())] {
            let error = result.expect_err("cancelled run");
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert_eq!(
                error.to_string(),
                "validation: run '".to_string() + handle.run_id() + "' has already been cancelled"
            );
        }
    }

    #[test]
    fn cancel_trips_the_token_and_mirrors_cancelled() {
        let monitor = Arc::new(RunMonitor::new());
        let hub = InteractionHub::new().with_monitor(Arc::clone(&monitor));
        let token = CancellationToken::new();
        let handle = hub.register(ActiveRunSpec {
            cancellation: Some(token.clone()),
            ..spec("assistant", Some("s-1"))
        });

        hub.cancel(handle.run_id()).unwrap();
        assert!(token.is_cancelled(), "the run's own token was tripped");
        assert!(handle.cancellation().is_cancelled());
        let info = hub.list_active().remove(0);
        assert_eq!(info.state, RunPhase::Cancelled);
        assert!(info.gate_closed);
        assert_eq!(
            monitor.snapshot(handle.run_id()).unwrap().phase,
            RunPhase::Cancelled
        );
        hub.cancel(handle.run_id()).unwrap();
    }

    #[test]
    fn list_active_is_sorted_by_run_id() {
        let hub = InteractionHub::new();
        for run_id in ["b", "a", "c"] {
            hub.register(ActiveRunSpec {
                run_id: Some(run_id.to_string()),
                ..ActiveRunSpec::new(run_id)
            });
        }
        let ids: Vec<String> = hub
            .list_active()
            .into_iter()
            .map(|info| info.run_id)
            .collect();
        assert_eq!(ids, vec!["a", "b", "c"]);
    }

    #[test]
    fn pending_views_default_to_empty() {
        let hub = InteractionHub::new();
        let handle = hub.register(spec("assistant", Some("s-1")));
        assert_eq!(hub.pending(handle.run_id()), Vec::<String>::new());
        assert_eq!(
            hub.pending_all(),
            vec![(handle.run_id().to_string(), Vec::new())]
        );
        assert!(hub.pending("run-x").is_empty());
    }

    #[tokio::test]
    async fn close_interactions_on_an_empty_hub_is_ok() {
        InteractionHub::new()
            .close_interactions("shutdown")
            .await
            .unwrap();
    }

    #[test]
    fn register_mirrors_into_the_monitor_without_duplicating_the_begin() {
        let monitor = Arc::new(RunMonitor::new());
        let hub = InteractionHub::new().with_monitor(Arc::clone(&monitor));
        assert!(hub.monitor().is_some());

        let handle = hub.register(ActiveRunSpec {
            team_name: Some("crew".to_string()),
            ..spec("crew", Some("s-1"))
        });
        let status = monitor.snapshot(handle.run_id()).unwrap();
        assert_eq!(status.phase, RunPhase::Queued);
        assert_eq!(status.kind, RunKind::Team);
        assert_eq!(status.session_id.as_deref(), Some("s-1"));
        assert_eq!(monitor.history(handle.run_id()).unwrap().len(), 1);

        // Re-registering the same id adopts the tracked run (no second begin).
        let _again = hub.register(ActiveRunSpec {
            run_id: Some(handle.run_id().to_string()),
            ..spec("crew", Some("s-1"))
        });
        assert_eq!(monitor.history(handle.run_id()).unwrap().len(), 1);

        hub.pause(handle.run_id()).unwrap();
        hub.resume(handle.run_id()).unwrap();
        let phases: Vec<RunPhase> = monitor
            .history(handle.run_id())
            .unwrap()
            .into_iter()
            .map(|event| event.phase)
            .collect();
        assert_eq!(
            phases,
            vec![RunPhase::Queued, RunPhase::Paused, RunPhase::Running]
        );
    }

    #[test]
    fn unregister_finalises_a_live_monitor_run_and_drops_the_entry() {
        let monitor = Arc::new(RunMonitor::new());
        let hub = InteractionHub::new().with_monitor(Arc::clone(&monitor));
        let handle = hub.register(spec("assistant", Some("s-1")));
        hub.unregister(handle.run_id()).unwrap();

        assert!(hub.is_empty());
        let status = monitor.snapshot(handle.run_id()).unwrap();
        assert_eq!(status.phase, RunPhase::Completed);
        assert!(status.finished_at.is_some());
        assert!(monitor.list_active().is_empty());
    }

    #[test]
    fn monitor_mirror_failures_do_not_fail_hub_operations() {
        let monitor = Arc::new(RunMonitor::new());
        let hub = InteractionHub::new().with_monitor(Arc::clone(&monitor));
        let handle = hub.register(spec("assistant", Some("s-1")));
        monitor
            .finish(handle.run_id(), RunOutcome::completed())
            .unwrap();

        hub.pause(handle.run_id()).unwrap();
        assert_eq!(hub.list_active()[0].state, RunPhase::Paused);
        assert_eq!(
            monitor.snapshot(handle.run_id()).unwrap().phase,
            RunPhase::Completed,
            "the hub's state is authoritative"
        );

        hub.cancel(handle.run_id()).unwrap();
        assert_eq!(hub.list_active()[0].state, RunPhase::Cancelled);
        hub.unregister(handle.run_id()).unwrap();
        assert_eq!(
            monitor.snapshot(handle.run_id()).unwrap().phase,
            RunPhase::Completed
        );

        // A monitor that never tracked the run is not an error either.
        let cleared = Arc::new(RunMonitor::new());
        let hub = InteractionHub::new().with_monitor(cleared);
        let handle = hub.register(spec("assistant", None));
        hub.monitor().unwrap().clear();
        hub.unregister(handle.run_id()).unwrap();
    }

    #[test]
    fn attach_without_a_runtime_still_attaches_the_session() {
        let hub = InteractionHub::new();
        let handle = hub.register(spec("assistant", Some("s-1")));
        hub.attach_session(handle.run_id(), Arc::new(AgentSession::new("s-1")))
            .unwrap();
        assert_eq!(hub.len(), 1);
        // The id is not indexed (no watcher); the fallback finds no request.
        assert!(hub.pending(handle.run_id()).is_empty());
    }

    #[tokio::test]
    async fn interaction_round_trip_through_a_session() {
        let hub = InteractionHub::new();
        let handle = hub.register(spec("assistant", Some("s-1")));
        let run_id = handle.run_id().to_string();
        let session = Arc::new(AgentSession::new("s-1"));
        hub.attach_session(&run_id, Arc::clone(&session) as Arc<dyn Session>)
            .unwrap();

        let request = InteractionRequest::ask_user(json!({"prompt": "ok?"}));
        let request_id = request.id.clone();
        let task = {
            let session = Arc::clone(&session);
            tokio::spawn(async move { session.request_interaction(request).await })
        };

        wait_pending(&hub, &run_id, 1).await;
        assert_eq!(hub.pending(&run_id), vec![request_id.clone()]);
        assert_eq!(
            hub.pending_all(),
            vec![(run_id.clone(), vec![request_id.clone()])]
        );
        let info = hub.list_active().remove(0);
        assert_eq!(info.queued, 1);

        hub.answer(InteractionResponse::answer(
            request_id.clone(),
            json!("yes"),
        ))
        .await
        .unwrap();
        let response = tokio::time::timeout(WAIT, task)
            .await
            .expect("the request resolves before the deadline")
            .expect("the request task does not panic")
            .expect("the answer is accepted");
        assert_eq!(response.request_id, request_id);
        assert_eq!(response.decision, InteractionDecision::Answer);
        assert_eq!(response.value, json!("yes"));
        assert!(hub.pending(&run_id).is_empty());
        assert_eq!(hub.list_active()[0].queued, 0);

        let error = hub
            .answer(InteractionResponse::answer("missing", json!(1)))
            .await
            .expect_err("an unknown interaction id");
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert_eq!(
            error.to_string(),
            "not_found: interaction 'missing' is not pending on any active run"
        );
    }

    #[tokio::test]
    async fn close_interactions_cancels_a_waiting_request() {
        let hub = InteractionHub::new();
        let handle = hub.register(spec("assistant", Some("s-1")));
        let run_id = handle.run_id().to_string();
        let session = Arc::new(AgentSession::new("s-1"));
        hub.attach_session(&run_id, Arc::clone(&session) as Arc<dyn Session>)
            .unwrap();

        let request = InteractionRequest::ask_user(json!({"prompt": "again?"}));
        let request_id = request.id.clone();
        let task = {
            let session = Arc::clone(&session);
            tokio::spawn(async move { session.request_interaction(request).await })
        };
        wait_pending(&hub, &run_id, 1).await;

        hub.close_interactions("shutdown").await.unwrap();
        assert!(hub.pending(&run_id).is_empty());

        let response = tokio::time::timeout(WAIT, task)
            .await
            .expect("the request resolves before the deadline")
            .expect("the request task does not panic")
            .expect("the cancellation is delivered");
        assert_eq!(response.request_id, request_id);
        assert_eq!(response.decision, InteractionDecision::Cancel);
        assert_eq!(response.reason.as_deref(), Some("shutdown"));
    }
}
