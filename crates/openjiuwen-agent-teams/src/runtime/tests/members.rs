//! Member lifecycle through a manager-built team: spawn, workspace isolation, retirement and the
//! task board.
//!
//! These cases cross the manager → spawner → harness-factory → workspace-provider chain (the unit
//! tests build the agent directly) and the leader → task board → bus chain of `create_task`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use openjiuwen_common::error::ErrorKind;
use std::collections::BTreeSet;
use std::path::PathBuf;

use crate::schema::{MemberLifetime, MemberStatus, TaskGraphSpec, TaskStatus, TeamTopic};
use crate::storage::TeamStore;

use super::support::{blueprint, event_types, recorded, subscribe_observer, Fixture};

#[tokio::test]
async fn auto_start_all_gives_every_teammate_a_distinct_workspace() {
    let fixture = Fixture::new();
    let team = fixture
        .activate(
            &blueprint("alpha", "boss", &["dev-1", "dev-2"]),
            "session-a",
        )
        .await;

    team.auto_start_all().await.unwrap();

    for name in ["dev-1", "dev-2"] {
        let member = team.backend().member(name).expect("the row exists");
        assert_eq!(member.status(), MemberStatus::Ready, "{name}");
        assert!(member.runtime().is_some(), "{name} owns a brain");
    }
    assert_eq!(
        team.backend().member("boss").expect("leader row").status(),
        MemberStatus::Ready,
        "the manager already started the leader"
    );

    let contexts = fixture.factory.calls();
    assert_eq!(contexts.len(), 3, "the leader plus two teammates");
    let roots: BTreeSet<PathBuf> = contexts
        .iter()
        .map(|context| context.workspace_root.clone())
        .collect();
    assert_eq!(
        roots.len(),
        3,
        "every member owns a distinct workspace root: {contexts:?}"
    );

    let requests = fixture.workspace.requests();
    assert_eq!(requests.len(), 3);
    assert!(
        requests
            .iter()
            .all(|request| request.session_id == team.session_id()),
        "every member runs under the team session"
    );
}

#[tokio::test]
async fn spawning_a_started_teammate_is_idempotent() {
    let fixture = Fixture::new();
    let team = fixture
        .activate(&blueprint("alpha", "boss", &["dev-1"]), "session-a")
        .await;
    team.auto_start_all().await.unwrap();
    let built = fixture.factory.calls().len();

    team.spawn_member("dev-1").await.unwrap();

    assert_eq!(
        fixture.factory.calls().len(),
        built,
        "no second harness is built"
    );
    assert_eq!(
        team.backend().member("dev-1").expect("the row").status(),
        MemberStatus::Ready
    );
}

#[tokio::test]
async fn shutdown_member_settles_on_the_requested_lifetime() {
    let fixture = Fixture::new();
    let team = fixture
        .activate(
            &blueprint("alpha", "boss", &["dev-1", "dev-2"]),
            "session-a",
        )
        .await;
    team.auto_start_all().await.unwrap();

    let dormant = team
        .shutdown_member("dev-1", MemberLifetime::Dormant, false)
        .await
        .unwrap();
    assert_eq!(dormant, MemberStatus::Dormant);
    assert_eq!(
        team.backend().member("dev-1").expect("the row").status(),
        MemberStatus::Dormant
    );

    let shut_down = team
        .shutdown_member("dev-2", MemberLifetime::Shutdown, true)
        .await
        .unwrap();
    assert_eq!(shut_down, MemberStatus::ShutDown);
    let stored = fixture
        .store
        .member("alpha", "dev-2")
        .await
        .unwrap()
        .expect("the row is persisted");
    assert_eq!(stored.status, MemberStatus::ShutDown, "the store agrees");
}

#[tokio::test]
async fn spawning_an_unknown_member_is_not_found() {
    let fixture = Fixture::new();
    let team = fixture
        .activate(&blueprint("alpha", "boss", &["dev-1"]), "session-a")
        .await;
    let built = fixture.factory.calls().len();

    let error = team.spawn_member("ghost").await.unwrap_err();

    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(
        error
            .to_string()
            .contains("member 'ghost' not found in team 'alpha'"),
        "{error}"
    );
    assert_eq!(
        fixture.factory.calls().len(),
        built,
        "an unknown member builds nothing"
    );
}

#[tokio::test]
async fn create_task_persists_the_row_and_publishes_the_created_event() {
    let fixture = Fixture::new();
    let team = fixture
        .activate(&blueprint("alpha", "boss", &["dev-1"]), "session-a")
        .await;
    let recorder = subscribe_observer(team.backend().messager(), &[TeamTopic::Task]);

    let record = team
        .create_task(TaskGraphSpec {
            title: "ship it".to_owned(),
            content: "do the thing".to_owned(),
            ..TaskGraphSpec::default()
        })
        .await
        .unwrap();

    let detail = team
        .backend()
        .tasks()
        .detail(&record.task_id)
        .await
        .unwrap()
        .expect("the row is persisted");
    assert_eq!(detail.title, "ship it");
    assert_eq!(detail.content, "do the thing");
    assert_eq!(detail.status, TaskStatus::Pending);

    let types = event_types(&recorder);
    assert_eq!(
        types.first().map(String::as_str),
        Some("task_created"),
        "{types:?}"
    );
    assert!(
        recorded(&recorder)
            .iter()
            .all(|event| event.sender_id == "alpha"),
        "board events are stamped with the team"
    );
}
