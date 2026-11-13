//! Observability hooks and monitor events for every lifecycle fact the runtime emits.
//!
//! [`TeamSignals`](crate::runtime::observability_bind::TeamSignals) is the one fan-out point, and
//! the mapping it implements is what these tests pin:
//!
//! | call | observability hook | monitor event |
//! | --- | --- | --- |
//! | `team_started` | — (opens the root span only) | `team_status` |
//! | `team_finished` | — (closes the root span) | `team_stopped` |
//! | `member_spawned` / `member_status` | `on_member_spawned` | `member_spawned` / `member_status` |
//! | `member_stopped` | `on_member_stopped` | `member_stopped` |
//! | `round_started` | `on_round_started` | — |
//! | `round_finished` | `on_round_finished` + `on_turn_metrics` | `member_round` |
//! | `task_created` | — | `task_created` |
//! | `task_status` | `on_task_transition` | `task_status` |
//! | `delegation` | `on_delegation` | — |

use std::sync::{Arc, Weak};

use serde_json::json;

use crate::member::{MemberActivityRegistry, MemberObserver, TeamMember};
use crate::monitor::{BackpressurePolicy, MonitorEvent};
use crate::observability::{
    MemberObservableEvent, ObservabilityHooks, ObservableEvent, RecordedEvent, RecordingHooks,
    RoundObservableEvent, TurnMetrics,
};
use crate::runtime::observability_bind::TeamSignals;
use crate::runtime::observer::MemberStreamObserver;
use crate::schema::{
    MemberLifetime, MemberStatus, RoundOutcome, TaskGraphSpec, TaskStatus, TeamRole,
};

use super::{blueprint, monitor_kinds, position_of, take_monitor, wire};

/// Returns the member payload of `record`, when the callback carried one.
fn member_event(record: &RecordedEvent) -> Option<&MemberObservableEvent> {
    match &record.event {
        ObservableEvent::Member(event) => Some(event),
        _ => None,
    }
}

/// Returns the round payload of `record`, when the callback carried one.
fn round_event(record: &RecordedEvent) -> Option<&RoundObservableEvent> {
    match &record.event {
        ObservableEvent::Round(event) => Some(event),
        _ => None,
    }
}

/// Returns the turn-metrics payload of `record`, when the callback carried one.
fn metrics_event(record: &RecordedEvent) -> Option<&TurnMetrics> {
    match &record.event {
        ObservableEvent::Turn(metrics) => Some(metrics),
        _ => None,
    }
}

/// Builds a signals bundle over `hooks` for team `alpha`.
fn signals_for(hooks: &Arc<RecordingHooks>) -> TeamSignals {
    TeamSignals::new(
        "alpha",
        vec![Arc::clone(hooks) as Arc<dyn ObservabilityHooks>],
    )
}

#[tokio::test]
async fn team_start_and_finish_move_the_root_span() {
    let hooks = Arc::new(RecordingHooks::new());
    let wiring = wire(&blueprint("alpha", "boss", &["dev-1"]), |deps| {
        deps.with_observability_hook(Arc::clone(&hooks) as Arc<dyn ObservabilityHooks>)
    })
    .await;
    let agent = &wiring.agent;

    assert!(
        agent.signals().observability().root_span().is_none(),
        "a built team has no trace yet"
    );

    agent.start().await.unwrap();
    let root = agent
        .signals()
        .observability()
        .root_span()
        .expect("start opened no root span");
    assert_eq!(root.name, "team.alpha");
    assert_eq!(
        root.attribute("gen_ai.conversation_id"),
        Some(&json!(agent.session_id())),
        "the root span carries the team session"
    );

    // Documented mapping: team start/finish fire no dedicated hook. What the leader's start does
    // produce is the member status observation its harness reports (`idle` -> `ready`).
    let kinds = hooks.kinds();
    assert!(!kinds.is_empty(), "the leader's start reported nothing");
    assert!(
        kinds.iter().all(|kind| *kind == "member_spawned"),
        "team start must not fire a teammate-only callback: {kinds:?}"
    );
    let leader = hooks
        .events()
        .into_iter()
        .find(|record| member_event(record).is_some_and(|event| event.member_name == "boss"))
        .expect("the leader's start status was not recorded");
    assert_eq!(member_event(&leader).unwrap().role, TeamRole::Leader);

    agent.stop().await.unwrap();
    assert!(
        agent.signals().observability().root_span().is_none(),
        "stop closes the root span"
    );
}

#[tokio::test]
async fn member_spawn_status_and_stop_reach_both_sinks() {
    let hooks = Arc::new(RecordingHooks::new());
    let wiring = wire(&blueprint("alpha", "boss", &["dev-1"]), |deps| {
        deps.with_observability_hook(Arc::clone(&hooks) as Arc<dyn ObservabilityHooks>)
    })
    .await;
    let agent = &wiring.agent;
    let mut monitor = agent
        .monitor_stream(64, BackpressurePolicy::DropOldest)
        .expect("no monitor stream after the build");

    agent.spawn_member("dev-1").await.unwrap();
    let spawned = hooks
        .events()
        .into_iter()
        .find(|record| {
            record.kind == "member_spawned"
                && member_event(record).is_some_and(|event| {
                    event.member_name == "dev-1" && event.role == TeamRole::Teammate
                })
        })
        .expect("the teammate's spawn was not recorded");
    let payload = member_event(&spawned).unwrap();
    assert_eq!(payload.team_name, "alpha");
    assert_eq!(spawned.span.name, "member.dev-1");

    // A direct status signal carries the same hook (it is the one that carries every status) and
    // produces the monitor `member_status` event.
    hooks.clear();
    agent
        .signals()
        .member_status("dev-1", TeamRole::Teammate, MemberStatus::Busy);
    let busy = hooks
        .events()
        .into_iter()
        .find(|record| record.kind == "member_spawned")
        .expect("the busy status was not recorded");
    assert_eq!(member_event(&busy).unwrap().status, MemberStatus::Busy);
    assert_eq!(member_event(&busy).unwrap().role, TeamRole::Teammate);
    let observed = take_monitor(&mut monitor);
    assert!(
        observed.iter().any(|event| matches!(
            event,
            MonitorEvent::MemberStatus { member_name, status: MemberStatus::Busy, .. }
                if member_name == "dev-1"
        )),
        "{:?}",
        monitor_kinds(&observed)
    );

    hooks.clear();
    agent
        .shutdown_member("dev-1", MemberLifetime::Shutdown, true)
        .await
        .unwrap();
    let stopped = hooks
        .events()
        .into_iter()
        .find(|record| record.kind == "member_stopped")
        .expect("the member stop was not recorded");
    assert_eq!(
        member_event(&stopped).unwrap().status,
        MemberStatus::Stopped
    );
    let observed = take_monitor(&mut monitor);
    assert!(
        observed.iter().any(|event| matches!(
            event,
            MonitorEvent::MemberStopped { member_name, reason: Some(reason) }
                if member_name == "dev-1" && reason == "shutdown"
        )),
        "{:?}",
        monitor_kinds(&observed)
    );
}

#[tokio::test]
async fn rounds_reach_the_hooks_and_the_monitor_once() {
    let hooks = Arc::new(RecordingHooks::new());
    let signals = signals_for(&hooks);
    let mut monitor = signals.monitor_stream(64, BackpressurePolicy::DropOldest);
    let observer = MemberStreamObserver::new(
        Arc::new(MemberActivityRegistry::new()),
        Weak::<TeamMember>::new(),
        TeamRole::Teammate,
        "dev-1".to_owned(),
        signals.clone(),
        Arc::new(parking_lot::Mutex::new(Vec::new())),
    );

    observer.on_round(RoundOutcome::Started);
    observer.on_round(RoundOutcome::Finished);

    let events = hooks.events();
    assert_eq!(
        hooks.kinds(),
        vec!["round_started", "round_finished", "turn_metrics"],
        "one start, one finish and its metrics"
    );

    let started = round_event(&events[0]).expect("round_started carried no round payload");
    assert_eq!(started.round, 1, "rounds are one-based");
    assert_eq!(started.outcome, RoundOutcome::Started);
    assert_eq!(started.member_name, "dev-1");

    let finished = round_event(&events[1]).expect("round_finished carried no round payload");
    assert_eq!(finished.round, 1);
    assert_eq!(finished.outcome, RoundOutcome::Finished);

    let metrics = metrics_event(&events[2]).expect("turn_metrics carried no metrics payload");
    assert_eq!(metrics.turn, 1);
    assert_eq!(metrics.team_name, "alpha");
    assert_eq!(metrics.member_name, "dev-1");
    assert!(
        metrics.duration_ms <= 2_000,
        "the elapsed duration is measured, not fabricated: {}",
        metrics.duration_ms
    );

    let round_events = take_monitor(&mut monitor);
    assert_eq!(monitor_kinds(&round_events), vec!["member_round"]);
    assert!(
        matches!(
            &round_events[0],
            MonitorEvent::MemberRound { member_name, round: 1, outcome: RoundOutcome::Finished }
                if member_name == "dev-1"
        ),
        "{:?}",
        round_events[0]
    );
}

#[tokio::test]
async fn task_transitions_reach_the_hooks_and_the_monitor() {
    let hooks = Arc::new(RecordingHooks::new());
    let wiring = wire(&blueprint("alpha", "boss", &["dev-1"]), |deps| {
        deps.with_observability_hook(Arc::clone(&hooks) as Arc<dyn ObservabilityHooks>)
    })
    .await;
    let agent = &wiring.agent;
    let mut monitor = agent
        .monitor_stream(64, BackpressurePolicy::DropOldest)
        .expect("no monitor stream after the build");

    let task = agent
        .create_task(TaskGraphSpec {
            title: "ship it".to_owned(),
            content: "the body".to_owned(),
            ..TaskGraphSpec::default()
        })
        .await
        .unwrap();

    // The creation mutation publishes the creation itself plus the `pending` status it lands on.
    let created = take_monitor(&mut monitor);
    assert_eq!(
        created.first().map(MonitorEvent::kind),
        Some("task_created"),
        "{:?}",
        monitor_kinds(&created)
    );
    assert!(
        matches!(
            &created[0],
            MonitorEvent::TaskCreated { task_id, title }
                if task_id == &task.task_id && title == "ship it"
        ),
        "{:?}",
        created[0]
    );
    assert!(
        created[1..].iter().all(|event| matches!(
            event,
            MonitorEvent::TaskStatus {
                status: TaskStatus::Pending,
                ..
            }
        )),
        "{:?}",
        monitor_kinds(&created)
    );
    hooks.clear();

    let claim = agent
        .backend()
        .tasks()
        .claim(&task.task_id, "dev-1")
        .await
        .unwrap();
    assert!(claim.is_ok(), "{:?}", claim.reason());
    let mutation = agent
        .backend()
        .tasks()
        .complete(&task.task_id, "dev-1", "done")
        .await
        .unwrap();
    agent.backend().publish_mutation(&mutation).await.unwrap();

    let transition = hooks
        .events()
        .into_iter()
        .find(|record| record.kind == "task_transition")
        .expect("the completion transition was not recorded");
    let ObservableEvent::Task(payload) = &transition.event else {
        panic!("a task_transition must carry a task payload");
    };
    assert_eq!(payload.team_name, "alpha");
    assert_eq!(payload.task_id, task.task_id);
    assert_eq!(payload.from, TaskStatus::Pending);
    assert_eq!(payload.to, TaskStatus::Completed);
    assert_eq!(
        payload.actor.as_deref(),
        Some("alpha"),
        "board events are published by the team, not a member"
    );
    assert_eq!(transition.span.name, format!("task.{}", task.task_id));

    let changed = take_monitor(&mut monitor);
    assert_eq!(monitor_kinds(&changed), vec!["task_status"]);
    assert!(
        matches!(
            &changed[0],
            MonitorEvent::TaskStatus { task_id, status: TaskStatus::Completed, assignee }
                if task_id == &task.task_id && assignee.as_deref() == Some("dev-1")
        ),
        "{:?}",
        changed[0]
    );
}

#[tokio::test]
async fn delegation_is_recorded_with_its_task() {
    let hooks = Arc::new(RecordingHooks::new());
    let wiring = wire(&blueprint("alpha", "boss", &["dev-1"]), |deps| {
        deps.with_observability_hook(Arc::clone(&hooks) as Arc<dyn ObservabilityHooks>)
    })
    .await;

    wiring
        .agent
        .signals()
        .delegation("boss", "dev-1", "task-1", "allowed");

    let recorded = hooks
        .events()
        .into_iter()
        .find(|record| record.kind == "delegation")
        .expect("the delegation was not recorded");
    let ObservableEvent::Delegation(event) = &recorded.event else {
        panic!("a delegation must carry a delegation payload");
    };
    assert_eq!(event.team_name, "alpha");
    assert_eq!(event.from_member, "boss");
    assert_eq!(event.to_member, "dev-1");
    assert_eq!(event.task_id.as_deref(), Some("task-1"));
    assert_eq!(event.reason.as_deref(), Some("allowed"));
    assert_eq!(recorded.span.name, "member.boss");
}

#[tokio::test]
async fn one_monitor_stream_keeps_the_lifecycle_order() {
    let wiring = wire(&blueprint("alpha", "boss", &["dev-1"]), |deps| deps).await;
    let agent = &wiring.agent;
    let mut monitor = agent
        .monitor_stream(128, BackpressurePolicy::DropOldest)
        .expect("no monitor stream after the build");
    assert!(
        agent.signals().monitor_emitter().is_some(),
        "installing a stream must install the emitter"
    );

    agent.start().await.unwrap();
    agent.spawn_member("dev-1").await.unwrap();
    agent
        .shutdown_member("dev-1", MemberLifetime::Shutdown, true)
        .await
        .unwrap();
    agent.stop().await.unwrap();

    // Every signal is emitted synchronously, so the queue is complete once `stop` returned.
    let events = take_monitor(&mut monitor);
    let kinds = monitor_kinds(&events);

    let team_status = position_of(&events, "team_status").expect("no team_status");
    let spawned = position_of(&events, "member_spawned").expect("no member_spawned");
    let stopped = position_of(&events, "member_stopped").expect("no member_stopped");
    let team_stopped = position_of(&events, "team_stopped").expect("no team_stopped");

    assert!(team_status < spawned, "start precedes the spawn: {kinds:?}");
    assert!(spawned < stopped, "spawn precedes the stop: {kinds:?}");
    assert!(
        stopped < team_stopped,
        "member stops precede the team stop: {kinds:?}"
    );
    assert_eq!(
        kinds.last(),
        Some(&"team_stopped"),
        "the team stop is the last event: {kinds:?}"
    );
    assert!(
        events.iter().enumerate().any(|(index, event)| {
            index > spawned && index < stopped && event.kind() == "member_status"
        }),
        "the spawned member reports a status before it stops: {kinds:?}"
    );
    assert!(
        events.iter().any(|event| matches!(
            event,
            MonitorEvent::MemberStopped { member_name, reason: Some(reason) }
                if member_name == "dev-1" && reason == "shutdown"
        )),
        "{kinds:?}"
    );
}
