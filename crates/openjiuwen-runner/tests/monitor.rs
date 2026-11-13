//! Run status monitoring: the transition sequence, `list_active` ordering, the
//! retention bounds, the typed error paths, the broadcast stream, and the JSON
//! shape of a status and an event.
//!
//! Drives only the public `openjiuwen_runner::monitor` surface, exactly as a host
//! UI or a dashboard would. Ports the active-run status bookkeeping Python's
//! `team_runner.py` kept per run (component label, session id, status); Python had
//! no standalone status type, so the field-level contract is the Rust one
//! documented in `src/monitor`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::time::Duration;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_runner::monitor::{
    RunEvent, RunKind, RunMonitor, RunOutcome, RunPhase, RunStart, RunStatus, EVENT_HISTORY_LIMIT,
    LIVE_RUN_LIMIT,
};
use serde_json::{json, Value};
use tokio::sync::broadcast;
use tokio::sync::broadcast::error::{RecvError, TryRecvError};

/// Upper bound on a deterministic in-memory channel read.
const BOUND: Duration = Duration::from_secs(10);

/// Awaits the next published event, bounded by [`BOUND`].
async fn next_event(receiver: &mut broadcast::Receiver<RunEvent>) -> Result<RunEvent, RecvError> {
    tokio::time::timeout(BOUND, receiver.recv())
        .await
        .expect("the monitor publishes one event per transition")
}

/// The phases a run's retained history records, oldest first.
fn phases_of(monitor: &RunMonitor, run_id: &str) -> Vec<RunPhase> {
    monitor
        .history(run_id)
        .expect("a tracked run")
        .into_iter()
        .map(|event| event.phase)
        .collect()
}

// ---------------------------------------------------------------------------
// The transition sequence
// ---------------------------------------------------------------------------

// ports team_runner.py's per-run status record (label + session id + status)
#[test]
fn begin_update_pause_and_finish_record_the_whole_sequence() {
    let monitor = RunMonitor::new();
    let created = monitor.begin(RunStart::new(RunKind::Agent, "assistant").with_session("s-1"));
    assert_eq!(created.phase, RunPhase::Queued);
    assert!(created.started_at > 0, "admission is stamped: {created:?}");
    assert!(created.finished_at.is_none());
    assert!(created.error.is_none());

    let running = monitor
        .update(&created.run_id, RunPhase::Running, json!({"step": 1}))
        .expect("running");
    assert_eq!(running.phase, RunPhase::Running);
    assert!(
        running.finished_at.is_none(),
        "only `finish` stamps the end"
    );
    assert_eq!(running.started_at, created.started_at);

    let paused = monitor
        .update(
            &created.run_id,
            RunPhase::Paused,
            json!({"awaiting": "human"}),
        )
        .expect("paused");
    assert_eq!(paused.phase, RunPhase::Paused);
    assert!(paused.finished_at.is_none());

    let done = monitor
        .finish(
            &created.run_id,
            RunOutcome::completed().with_detail(json!({"steps": 2})),
        )
        .expect("finished");
    assert_eq!(done.phase, RunPhase::Completed);
    assert!(
        done.finished_at.is_some(),
        "the terminal transition stamps it"
    );
    assert!(done.finished_at >= Some(done.started_at));
    assert!(done.error.is_none());

    let history = monitor.history(&created.run_id).expect("history");
    assert_eq!(history.len(), 4, "one event per transition");
    for (event, phase) in history.iter().zip([
        RunPhase::Queued,
        RunPhase::Running,
        RunPhase::Paused,
        RunPhase::Completed,
    ]) {
        assert_eq!(event.run_id, created.run_id);
        assert_eq!(event.kind, RunKind::Agent);
        assert_eq!(event.label, "assistant");
        assert_eq!(event.phase, phase);
        assert_eq!(event.session_id.as_deref(), Some("s-1"));
    }
    assert!(
        history
            .windows(2)
            .all(|pair| pair[0].at_ms <= pair[1].at_ms),
        "history is oldest first: {history:?}"
    );
}

// ---------------------------------------------------------------------------
// The active list
// ---------------------------------------------------------------------------

#[test]
fn list_active_holds_only_the_live_runs_in_a_stable_order() {
    let monitor = RunMonitor::new();
    let first = monitor.begin(RunStart::new(RunKind::Agent, "first"));
    let second = monitor.begin(RunStart::new(RunKind::Workflow, "second"));
    let third = monitor.begin(RunStart::new(RunKind::Team, "third"));

    let active = monitor.list_active();
    assert_eq!(active.len(), 3);
    assert!(
        active
            .windows(2)
            .all(|pair| (pair[0].started_at, &pair[0].run_id)
                <= (pair[1].started_at, &pair[1].run_id)),
        "sorted by (started_at, run_id): {active:?}"
    );

    monitor
        .finish(&first.run_id, RunOutcome::completed())
        .expect("finish");

    let remaining = monitor.list_active();
    let ids: Vec<String> = remaining
        .iter()
        .map(|status| status.run_id.clone())
        .collect();
    let expected = if (second.started_at, &second.run_id) <= (third.started_at, &third.run_id) {
        vec![second.run_id.clone(), third.run_id.clone()]
    } else {
        vec![third.run_id.clone(), second.run_id.clone()]
    };
    assert_eq!(ids, expected, "the finished run left the active list");
    assert!(
        remaining.iter().all(|status| !status.is_terminal()),
        "only non-terminal runs are listed"
    );
}

// ---------------------------------------------------------------------------
// Snapshots and history
// ---------------------------------------------------------------------------

#[test]
fn snapshot_serves_a_live_and_a_retained_finished_run() {
    let monitor = RunMonitor::new();
    let live = monitor.begin(RunStart::new(RunKind::Agent, "assistant").with_session("s-live"));
    monitor
        .update(&live.run_id, RunPhase::Running, json!({"step": 1}))
        .expect("running");

    let snapshot = monitor.snapshot(&live.run_id).expect("live snapshot");
    assert_eq!(snapshot.phase, RunPhase::Running);
    assert_eq!(snapshot.session_id.as_deref(), Some("s-live"));
    assert!(snapshot.finished_at.is_none());
    assert_eq!(
        phases_of(&monitor, &live.run_id),
        vec![RunPhase::Queued, RunPhase::Running]
    );

    let done = monitor.begin(RunStart::new(RunKind::Team, "crew"));
    monitor
        .finish(
            &done.run_id,
            RunOutcome::cancelled().with_detail(json!({"reason": "shutdown"})),
        )
        .expect("finish");

    let snapshot = monitor.snapshot(&done.run_id).expect("retained snapshot");
    assert_eq!(snapshot.kind, RunKind::Team);
    assert_eq!(snapshot.phase, RunPhase::Cancelled);
    assert!(snapshot.is_terminal());
    assert!(snapshot.finished_at.is_some());
    let history = monitor.history(&done.run_id).expect("retained history");
    assert_eq!(
        history.iter().map(|event| event.phase).collect::<Vec<_>>(),
        vec![RunPhase::Queued, RunPhase::Cancelled]
    );
    assert_eq!(history[1].detail, json!({"reason": "shutdown"}));
}

// ---------------------------------------------------------------------------
// Retention bounds
// ---------------------------------------------------------------------------

#[test]
fn history_keeps_the_newest_events_of_a_run() {
    let monitor = RunMonitor::new();
    let created = monitor.begin(RunStart::new(RunKind::Workflow, "wf"));
    let transitions = EVENT_HISTORY_LIMIT + 4;
    for step in 0..transitions {
        monitor
            .update(&created.run_id, RunPhase::Running, json!({"step": step}))
            .expect("update");
    }

    let history = monitor.history(&created.run_id).expect("history");
    assert_eq!(
        history.len(),
        EVENT_HISTORY_LIMIT,
        "the per-run history is bounded"
    );
    assert_eq!(
        history[0].phase,
        RunPhase::Running,
        "the oldest entries (including the initial `Queued`) were dropped"
    );
    let dropped = transitions + 1 - EVENT_HISTORY_LIMIT;
    assert_eq!(
        history[0].detail,
        json!({"step": dropped - 1}),
        "the oldest retained event is the ({dropped})th transition"
    );
    assert_eq!(
        history[EVENT_HISTORY_LIMIT - 1].detail,
        json!({"step": transitions - 1}),
        "the newest transition is retained"
    );
}

#[test]
fn finished_runs_are_retained_up_to_the_live_run_limit() {
    let monitor = RunMonitor::new();
    let mut finished = Vec::new();
    for index in 0..=LIVE_RUN_LIMIT {
        let status = monitor.begin(RunStart::new(RunKind::Agent, format!("run-{index}")));
        monitor
            .finish(&status.run_id, RunOutcome::completed())
            .expect("finish");
        finished.push(status.run_id);
    }

    let evicted = monitor
        .snapshot(&finished[0])
        .expect_err("the oldest completion is evicted");
    assert_eq!(evicted.kind(), ErrorKind::NotFound);
    assert!(
        monitor.history(&finished[0]).is_err(),
        "the evicted run's history goes with it"
    );
    assert!(
        monitor.snapshot(&finished[1]).is_ok(),
        "only as many runs as the limit are evicted"
    );
    let newest = monitor
        .snapshot(&finished[LIVE_RUN_LIMIT])
        .expect("the newest completion is retained");
    assert_eq!(newest.phase, RunPhase::Completed);
    assert!(newest.finished_at.is_some());
    assert_eq!(monitor.list_active(), Vec::new(), "every run finished");
}

// ---------------------------------------------------------------------------
// Typed error paths
// ---------------------------------------------------------------------------

#[test]
fn unknown_ids_and_terminal_transitions_report_typed_errors() {
    let monitor = RunMonitor::new();

    for error in [
        monitor.snapshot("run-missing").unwrap_err(),
        monitor.history("run-missing").unwrap_err(),
        monitor
            .update("run-missing", RunPhase::Running, Value::Null)
            .unwrap_err(),
        monitor
            .finish("run-missing", RunOutcome::completed())
            .unwrap_err(),
    ] {
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert_eq!(
            error.to_string(),
            "not_found: run 'run-missing' is not tracked"
        );
    }

    let done = monitor.begin(RunStart::new(RunKind::Agent, "assistant"));
    monitor
        .finish(&done.run_id, RunOutcome::completed())
        .expect("finish");

    let double = monitor
        .finish(&done.run_id, RunOutcome::cancelled())
        .unwrap_err();
    assert_eq!(double.kind(), ErrorKind::Validation);
    assert!(double.to_string().contains("has already finished"));
    let late = monitor
        .update(&done.run_id, RunPhase::Running, Value::Null)
        .unwrap_err();
    assert_eq!(late.kind(), ErrorKind::Validation);
    assert_eq!(
        monitor.snapshot(&done.run_id).expect("snapshot").phase,
        RunPhase::Completed,
        "a rejected transition changes nothing"
    );

    // A terminal phase must go through `finish`: `update` only advances live
    // runs, so a terminal update is rejected without changing the run.
    let live = monitor.begin(RunStart::new(RunKind::Workflow, "wf"));
    for phase in [RunPhase::Completed, RunPhase::Failed, RunPhase::Cancelled] {
        let error = monitor
            .update(&live.run_id, phase, json!({"step": "llm"}))
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("must be finished with `finish`"));
    }
    let status = monitor.snapshot(&live.run_id).expect("still live");
    assert_eq!(status.phase, RunPhase::Queued);
    assert!(status.finished_at.is_none());
    assert_eq!(
        monitor.list_active().len(),
        1,
        "a rejected terminal update keeps the run live"
    );
    assert_eq!(monitor.history(&live.run_id).expect("history").len(), 1);
}

// ---------------------------------------------------------------------------
// The event channel
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_subscriber_receives_one_event_per_transition_in_order() {
    let monitor = RunMonitor::new();
    let mut receiver = monitor.subscribe();
    let created = monitor.begin(RunStart::new(RunKind::Agent, "assistant"));
    monitor
        .update(&created.run_id, RunPhase::Running, Value::Null)
        .expect("running");
    monitor
        .update(&created.run_id, RunPhase::Paused, Value::Null)
        .expect("paused");
    monitor
        .finish(&created.run_id, RunOutcome::completed())
        .expect("finish");

    let mut phases = Vec::new();
    for _ in 0..4 {
        let event = next_event(&mut receiver).await.expect("a published event");
        assert_eq!(event.run_id, created.run_id);
        assert_eq!(event.kind, RunKind::Agent);
        assert_eq!(event.label, "assistant");
        assert!(event.at_ms > 0);
        phases.push(event.phase);
    }
    assert_eq!(
        phases,
        vec![
            RunPhase::Queued,
            RunPhase::Running,
            RunPhase::Paused,
            RunPhase::Completed
        ],
        "one event per transition, in transition order"
    );
    assert_eq!(
        receiver.try_recv(),
        Err(TryRecvError::Empty),
        "no extra event was published"
    );
}

#[tokio::test]
async fn a_lagged_subscriber_is_told_and_then_keeps_receiving() {
    let monitor = RunMonitor::with_capacity(1);
    let mut receiver = monitor.subscribe();
    let created = monitor.begin(RunStart::new(RunKind::Agent, "assistant"));
    monitor
        .update(&created.run_id, RunPhase::Running, Value::Null)
        .expect("running");
    monitor
        .update(&created.run_id, RunPhase::Paused, Value::Null)
        .expect("paused");
    monitor
        .update(&created.run_id, RunPhase::Running, json!({"step": 2}))
        .expect("running again");

    match next_event(&mut receiver).await {
        Err(RecvError::Lagged(dropped)) => assert!(dropped >= 1, "frames were dropped"),
        other => panic!("expected a Lagged error, got {other:?}"),
    }
    let buffered = next_event(&mut receiver)
        .await
        .expect("the oldest buffered frame");
    assert_eq!(
        buffered.phase,
        RunPhase::Running,
        "the capacity is one frame, so the newest one survived"
    );

    monitor
        .finish(&created.run_id, RunOutcome::completed())
        .expect("finish");
    let event = next_event(&mut receiver)
        .await
        .expect("events published after the lag still arrive");
    assert_eq!(event.phase, RunPhase::Completed);
    assert_eq!(event.run_id, created.run_id);
}

#[test]
fn clear_drops_every_status_and_history() {
    let monitor = RunMonitor::new();
    let live = monitor.begin(RunStart::new(RunKind::Agent, "assistant"));
    let done = monitor.begin(RunStart::new(RunKind::Workflow, "wf"));
    monitor
        .finish(&done.run_id, RunOutcome::completed())
        .expect("finish");

    monitor.clear();

    assert_eq!(monitor.list_active(), Vec::new(), "clear empties the table");
    for run_id in [&live.run_id, &done.run_id] {
        let error = monitor.snapshot(run_id).expect_err("cleared");
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert!(monitor.history(run_id).is_err());
    }

    let next = monitor.begin(RunStart::new(RunKind::Team, "crew"));
    assert_eq!(monitor.list_active().len(), 1);
    assert_eq!(monitor.list_active()[0].run_id, next.run_id);
}

// ---------------------------------------------------------------------------
// JSON shape
// ---------------------------------------------------------------------------

#[test]
fn status_and_event_round_trip_as_snake_case_json() {
    let status = RunStatus {
        run_id: "run-1".to_string(),
        kind: RunKind::Team,
        label: "crew".to_string(),
        phase: RunPhase::Completed,
        session_id: Some("s-1".to_string()),
        started_at: 1_700_000_000_000,
        finished_at: Some(1_700_000_000_500),
        error: None,
        detail: json!({"steps": 2}),
    };
    let value = serde_json::to_value(&status).expect("serializable");
    assert_eq!(value["run_id"], json!("run-1"));
    assert_eq!(value["kind"], json!("team"));
    assert_eq!(value["label"], json!("crew"));
    assert_eq!(value["phase"], json!("completed"));
    assert_eq!(value["session_id"], json!("s-1"));
    assert_eq!(value["started_at"], json!(1_700_000_000_000_i64));
    assert_eq!(value["finished_at"], json!(1_700_000_000_500_i64));
    assert_eq!(value["error"], Value::Null);
    let mut keys: Vec<&str> = value
        .as_object()
        .expect("an object")
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        vec![
            "detail",
            "error",
            "finished_at",
            "kind",
            "label",
            "phase",
            "run_id",
            "session_id",
            "started_at",
        ],
        "every field is snake_case"
    );
    let back: RunStatus = serde_json::from_value(value).expect("round trip");
    assert_eq!(back, status);

    let event = RunEvent {
        run_id: "run-1".to_string(),
        kind: RunKind::Workflow,
        label: "wf".to_string(),
        phase: RunPhase::Running,
        session_id: None,
        at_ms: 1_700_000_000_250,
        detail: json!({"step": 1}),
    };
    let value = serde_json::to_value(&event).expect("serializable");
    assert_eq!(value["run_id"], json!("run-1"));
    assert_eq!(value["phase"], json!("running"));
    assert_eq!(value["session_id"], Value::Null);
    assert_eq!(value["at_ms"], json!(1_700_000_000_250_i64));
    let back: RunEvent = serde_json::from_value(value).expect("round trip");
    assert_eq!(back, event);
}
