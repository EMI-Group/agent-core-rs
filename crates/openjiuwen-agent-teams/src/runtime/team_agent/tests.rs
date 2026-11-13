use super::test_support::{blueprint, built, deps, ScriptedFactory};
use super::{ensure_leader_role, SpecBuild, TeamAgentSpec};
use crate::interaction::{ExternalTeamEvent, HumanAgentMessage, InteractPayload, OperatorMessage};
use crate::member::MemberLifetime;
use crate::schema::{MemberStatus, TaskGraphSpec, TeamRole};
use openjiuwen_common::error::ErrorKind;
use openjiuwen_harness_protocol::HarnessState;
use serde_json::Value;

/// Builds a two-teammate blueprint for the shared tests.
fn two_teammates() -> TeamAgentSpec {
    blueprint("alpha", "boss", &["dev-1", "dev-2"])
}

#[tokio::test]
async fn build_produces_the_leader_handle_and_the_roster() {
    let (agent, _factory, _workspace) = built(&two_teammates()).await;

    assert_eq!(agent.card().name, "alpha_boss");
    assert_eq!(agent.role(), TeamRole::Leader);
    assert_eq!(agent.member_name(), "boss");
    assert_eq!(agent.leader_member_name(), "boss");
    assert_eq!(agent.team_name(), "alpha");
    assert_eq!(agent.spec().team_name, "alpha");
    assert!(agent.session_id().starts_with("session_"));
    assert_eq!(agent.roster(), vec!["boss", "dev-1", "dev-2"]);
    assert!(!agent.is_started());

    let leader = agent.backend().member("boss").unwrap();
    assert_eq!(leader.role(), TeamRole::Leader);
    assert_eq!(leader.status(), MemberStatus::Unstarted);
    assert_eq!(leader.record().prompt, "prompt for Leader");

    for name in ["dev-1", "dev-2"] {
        let member = agent.backend().member(name).unwrap();
        assert_eq!(member.role(), TeamRole::Teammate);
        assert_eq!(member.status(), MemberStatus::Unstarted);
        assert_eq!(member.record().prompt, format!("prompt for {name}"));
        assert!(
            member.runtime().is_none(),
            "teammates start without a brain"
        );
    }

    let card = agent.team_card();
    assert_eq!(card.id, "alpha");
    assert_eq!(card.name, "alpha");
    assert_eq!(card.agent_cards.len(), 3);
    assert_eq!(card.agent_cards[0].name, "alpha_boss");
    assert_eq!(agent.runtime().session_id(), agent.session_id());
}

#[tokio::test]
async fn build_requires_the_leader_blueprint() {
    let spec = TeamAgentSpec {
        team_name: "alpha".to_owned(),
        ..TeamAgentSpec::default()
    };
    let error = spec
        .build_with(deps(
            &ScriptedFactory::new(),
            &super::test_support::RecordingWorkspaceProvider::new(),
        ))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error
            .to_string()
            .contains("agents dict must contain a 'leader' key"),
        "{error}"
    );
}

#[tokio::test]
async fn language_comes_from_the_deps_else_the_blueprint() {
    let factory = ScriptedFactory::new();
    let workspace = super::test_support::RecordingWorkspaceProvider::new();
    let agent = blueprint("alpha", "boss", &[])
        .build_with(deps(&factory, &workspace).with_language("en"))
        .await
        .unwrap();
    assert_eq!(agent.language(), "en");
    assert_eq!(agent.spec().language.as_deref(), Some("en"));

    let mut declared = blueprint("beta", "boss", &[]);
    declared.language = Some("cf".to_owned());
    let agent = declared
        .build_with(deps(&factory, &workspace))
        .await
        .unwrap();
    assert_eq!(agent.language(), "cf");

    let agent = blueprint("gamma", "boss", &[])
        .build_with(deps(&factory, &workspace).with_language("bogus"))
        .await
        .unwrap();
    assert_eq!(agent.language(), "cn", "an unknown language falls back");
}

#[tokio::test]
async fn start_is_idempotent_and_stop_settles_the_leader_row() {
    let (agent, _factory, _workspace) = built(&two_teammates()).await;
    agent.start().await.unwrap();
    agent.start().await.unwrap();
    assert!(agent.is_started());
    assert_eq!(agent.state(), HarnessState::Idle);
    assert_eq!(
        agent.backend().member("boss").unwrap().status(),
        MemberStatus::Ready
    );

    agent.stop().await.unwrap();
    agent.stop().await.unwrap();
    assert!(!agent.is_started());
    assert_eq!(
        agent.backend().member("boss").unwrap().status(),
        MemberStatus::ShutDown
    );
    assert!(agent.gate().is_closed());
}

#[tokio::test]
async fn spawning_an_unknown_member_is_not_found() {
    let (agent, _factory, _workspace) = built(&two_teammates()).await;
    let error = agent.spawn_member("ghost").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(error
        .to_string()
        .contains("member 'ghost' not found in team 'alpha'"));
}

#[tokio::test]
async fn spawning_a_teammate_builds_and_starts_its_runtime() {
    let (agent, factory, _workspace) = built(&two_teammates()).await;
    agent.spawn_member("dev-1").await.unwrap();

    let member = agent.backend().member("dev-1").unwrap();
    assert_eq!(member.status(), MemberStatus::Ready);
    assert!(member.runtime().is_some());
    let calls: Vec<String> = factory
        .harness("dev-1")
        .unwrap()
        .calls()
        .iter()
        .map(|call| format!("{call:?}"))
        .collect();
    assert!(calls.contains(&"Start".to_owned()), "{calls:?}");

    // A second spawn of a running member is a no-op.
    agent.spawn_member("dev-1").await.unwrap();
    assert_eq!(
        agent.backend().member("dev-1").unwrap().status(),
        MemberStatus::Ready
    );
}

#[tokio::test]
async fn a_failed_spawn_keeps_the_row_unstarted() {
    let (agent, factory, _workspace) = built(&two_teammates()).await;
    factory.failing_for("dev-1");
    let error = agent.spawn_member("dev-1").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Config);
    assert_eq!(
        agent.backend().member("dev-1").unwrap().status(),
        MemberStatus::Unstarted
    );
    assert!(agent.backend().member("dev-1").unwrap().runtime().is_none());
}

#[tokio::test]
async fn shutdown_member_settles_on_the_lifetime_target() {
    let (agent, _factory, _workspace) = built(&two_teammates()).await;
    agent.spawn_member("dev-1").await.unwrap();
    let status = agent
        .shutdown_member("dev-1", MemberLifetime::Shutdown, true)
        .await
        .unwrap();
    assert_eq!(status, MemberStatus::ShutDown);

    agent.spawn_member("dev-2").await.unwrap();
    let status = agent
        .shutdown_member("dev-2", MemberLifetime::Dormant, false)
        .await
        .unwrap();
    assert_eq!(status, MemberStatus::Dormant);

    let error = agent
        .shutdown_member("ghost", MemberLifetime::Shutdown, true)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
}

#[tokio::test]
async fn auto_start_all_starts_every_unstarted_teammate() {
    let (agent, _factory, _workspace) = built(&two_teammates()).await;
    agent.auto_start_all().await.unwrap();
    for name in ["dev-1", "dev-2"] {
        assert_eq!(
            agent.backend().member(name).unwrap().status(),
            MemberStatus::Ready,
            "{name}"
        );
    }
    assert_eq!(
        agent.backend().member("boss").unwrap().status(),
        MemberStatus::Unstarted,
        "auto_start_all never starts the leader"
    );
}

#[tokio::test]
async fn every_member_gets_its_own_workspace_root() {
    let (agent, factory, workspace) = built(&two_teammates()).await;
    agent.spawn_member("dev-1").await.unwrap();

    let contexts = factory.calls();
    assert_eq!(
        contexts.len(),
        2,
        "one harness for the leader, one for dev-1"
    );
    let leader_root = contexts[0].workspace_root.clone();
    let teammate_root = contexts[1].workspace_root.clone();
    assert_ne!(leader_root, teammate_root);
    assert!(leader_root.ends_with("boss"), "{leader_root:?}");
    assert!(teammate_root.ends_with("dev-1"), "{teammate_root:?}");

    let requests = workspace.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(
        requests[0].session_id, requests[1].session_id,
        "both members run under the team session"
    );
}

#[tokio::test]
async fn stop_releases_every_member_workspace_exactly_once() {
    let (agent, _factory, workspace) = built(&two_teammates()).await;
    agent.start().await.unwrap();
    agent.auto_start_all().await.unwrap();
    assert!(
        workspace.releases().is_empty(),
        "nothing is released while the team is live"
    );

    agent.stop().await.unwrap();

    let mut released = workspace.released_names();
    released.sort();
    assert_eq!(
        released,
        ["boss", "dev-1", "dev-2"],
        "the leader and every spawned member is released"
    );

    agent.stop().await.unwrap();
    assert_eq!(
        workspace.released_names().len(),
        3,
        "a second stop releases nothing"
    );
}

#[tokio::test]
async fn shutdown_member_releases_only_that_members_workspace() {
    let (agent, _factory, workspace) = built(&two_teammates()).await;
    agent.start().await.unwrap();
    agent.auto_start_all().await.unwrap();

    agent
        .shutdown_member("dev-1", MemberLifetime::Shutdown, true)
        .await
        .unwrap();
    assert_eq!(
        workspace.released_names(),
        ["dev-1"],
        "only the retired member's workspace is handed back"
    );

    agent
        .shutdown_member("dev-1", MemberLifetime::Shutdown, true)
        .await
        .unwrap();
    assert_eq!(
        workspace.released_names(),
        ["dev-1"],
        "a second shutdown of the same member releases nothing"
    );

    agent.stop().await.unwrap();
    let mut released = workspace.released_names();
    released.sort();
    assert_eq!(
        released,
        ["boss", "dev-1", "dev-2"],
        "stop releases the members that are still retained"
    );
}

#[tokio::test]
async fn a_failed_release_is_logged_and_does_not_fail_teardown() {
    let (agent, _factory, workspace) = built(&two_teammates()).await;
    agent.start().await.unwrap();
    agent.auto_start_all().await.unwrap();
    workspace.fail_releases();

    agent
        .shutdown_member("dev-1", MemberLifetime::Shutdown, true)
        .await
        .expect("a release failure does not fail shutdown_member");
    agent
        .stop()
        .await
        .expect("a release failure does not fail stop");
    assert!(workspace.releases().is_empty(), "every release failed");
    assert_eq!(
        agent.backend().member("boss").unwrap().status(),
        MemberStatus::ShutDown,
        "the leader row still settles"
    );
}

#[tokio::test]
async fn a_failed_spawn_hands_its_workspace_back() {
    let (agent, factory, workspace) = built(&two_teammates()).await;
    agent.start().await.unwrap();
    factory.failing_for("dev-1");

    let error = agent.spawn_member("dev-1").await.unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Config);
    assert_eq!(
        workspace.released_names(),
        ["dev-1"],
        "the workspace of a member that never got a runtime is released at once"
    );
    agent.stop().await.unwrap();
    assert_eq!(
        workspace
            .released_names()
            .iter()
            .filter(|name| name.as_str() == "dev-1")
            .count(),
        1,
        "the failed member's workspace is released exactly once"
    );
}

#[tokio::test]
async fn create_task_is_leader_only_and_publishes_its_events() {
    let error = ensure_leader_role(TeamRole::Teammate).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error
            .to_string()
            .contains("only the leader can create tasks"),
        "{error}"
    );
    assert!(ensure_leader_role(TeamRole::Leader).is_ok());

    let (agent, _factory, _workspace) = built(&two_teammates()).await;
    let record = agent
        .create_task(TaskGraphSpec {
            title: "ship it".to_owned(),
            content: "do the thing".to_owned(),
            ..TaskGraphSpec::default()
        })
        .await
        .unwrap();
    assert_eq!(record.title, "ship it");
    assert!(!agent.backend().tasks().is_settled().await.unwrap());
}

#[tokio::test]
async fn deliver_routes_every_channel() {
    let (agent, _factory, _workspace) = built(&two_teammates()).await;

    let human = agent
        .deliver(InteractPayload::HumanAgent(HumanAgentMessage {
            human_name: "alice".to_owned(),
            target: None,
            body: "the build is green".to_owned(),
            sender: Some("alice".to_owned()),
        }))
        .await;
    assert!(human.ok, "{human:?}");
    assert!(human.message_id.is_some(), "the delivered row is returned");
    assert!(!agent.inbox().is_enabled(), "HITL is off by default");

    let operator = agent
        .deliver(InteractPayload::Operator(OperatorMessage {
            target: None,
            body: "status?".to_owned(),
            sender: None,
        }))
        .await;
    assert_eq!(operator.reason.as_deref(), Some("not_active"));

    let external = agent
        .deliver(InteractPayload::ExternalTeam(ExternalTeamEvent {
            event: Value::Null,
        }))
        .await;
    assert_eq!(
        external.reason.as_deref(),
        Some("unsupported:external_team")
    );
}

#[tokio::test]
async fn interact_reports_the_gate_state() {
    let (agent, _factory, _workspace) = built(&two_teammates()).await;
    let payload = InteractPayload::GodView(crate::interaction::GodViewMessage {
        body: "hello".to_owned(),
        sender: None,
    });
    assert_eq!(
        agent.interact(payload.clone()).await.reason.as_deref(),
        Some("not_active")
    );
    agent.start().await.unwrap();
    assert!(agent.interact(payload.clone()).await.ok);
    agent.stop().await.unwrap();
    assert_eq!(
        agent.interact(payload).await.reason.as_deref(),
        Some("gate_closed")
    );
}
