//! Spec parsing, validation, build and run-admission behaviour, through the public API.
//!
//! The suite pins four contracts a caller of `openjiuwen-agent-teams` relies on:
//!
//! 1. a full Python-shaped spec parses from JSON, tolerating unknown fields, and every omitted
//!    knob lands on its Python default;
//! 2. [`TeamAgentSpec::validate`] rejects the same blueprints Python rejects, with the same
//!    error kind and message text;
//! 3. [`SpecBuild::build_with`] wires the leader and the roster exactly once, handing the harness
//!    factory the leader's allocated workspace, and resolves the prompt language from the
//!    dependencies before the spec;
//! 4. [`TeamRuntimeManager::activate`] admits a create, a same-session attach and a persistent
//!    cross-session resume, and rejects a temporary team's cross-session request.
//!
//! Everything is offline: the fakes in `support` answer every member input with one scripted
//! observation triple and allocate member workspaces under one temporary directory.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![allow(clippy::missing_panics_doc)]

mod support;

use std::sync::Arc;

use openjiuwen_agent_teams::runtime::SpecBuild;
use openjiuwen_agent_teams::schema::{
    BootstrapMode, ConversationMode, DispatchMode, MemberMode, MemberStatus, SpawnMode,
    TeamAgentSpec, TeamLifecycle, TeamRole,
};
use openjiuwen_agent_teams::storage::{InMemoryTeamStore, TeamStore};
use openjiuwen_common::error::ErrorKind;
use serde_json::{json, Value};

use support::{
    build_deps, sample_spec, sample_spec_verified, FakeHarnessFactory, FakeWorkspaceProvider,
    Fixture,
};

/// A Python-shaped blueprint with every required field, two explicit defaults and unknown fields
/// at three levels.
fn full_spec_json() -> Value {
    json!({
        "team_name": "beta",
        "team_display_name": "Beta Team",
        "team_desc": "a team parsed from JSON",
        "agents": {
            "leader": {
                "card": {"name": "Beta Lead", "description": "leads beta"},
                "system_prompt": "lead the beta team",
            },
            "teammate": {
                "card": {"name": "Beta Teammate"},
                "system_prompt": "do the beta work",
            },
        },
        "leader": {
            "member_name": "team_leader",
            "display_name": "Team Leader",
            "desc": "leads the team",
            "unknown_leader_knob": {"nested": true},
        },
        "predefined_members": [
            {
                "role_type": "teammate",
                "member_name": "dev-1",
                "display_name": "Dev One",
                "desc": "the developer",
                "prompt": "you are dev-1",
            },
        ],
        "transport": {"kind": "inprocess"},
        "storage": {"kind": "memory"},
        "workspace": {"enabled": false, "unknown_workspace_knob": 7},
        "unknown_top_level_knob": ["ignored", "verbatim"],
    })
}

/// Builds a blueprint with a leader blueprint and one predefined teammate `name`.
fn spec_with_predefined(name: &str) -> TeamAgentSpec {
    let mut spec = sample_spec();
    spec.predefined_members = vec![
        openjiuwen_agent_teams::schema::PredefinedMemberSpec::Teammate(
            openjiuwen_agent_teams::schema::MemberSpecBase {
                member_name: name.to_owned(),
                display_name: name.to_owned(),
                desc: String::new(),
                prompt: String::new(),
                model_name: None,
            },
        ),
    ];
    spec
}

#[test]
fn full_spec_parses_and_leaves_defaults_intact() {
    let spec: TeamAgentSpec =
        serde_json::from_value(full_spec_json()).expect("the full blueprint must parse");

    assert_eq!(spec.team_name, "beta");
    assert_eq!(spec.team_display_name.as_deref(), Some("Beta Team"));
    assert_eq!(spec.team_desc.as_deref(), Some("a team parsed from JSON"));

    // Agents, predefined members and the two seams survive unchanged.
    assert_eq!(spec.agents.len(), 2);
    assert!(spec.agents.contains_key("leader"));
    assert_ne!(
        spec.agents["leader"].system_prompt.as_deref(),
        Some(""),
        "the leader prompt must survive the parse"
    );
    assert_eq!(spec.predefined_members.len(), 1);
    assert_eq!(spec.predefined_members[0].member_name(), "dev-1");
    assert_eq!(spec.predefined_members[0].role_type(), TeamRole::Teammate);
    assert_eq!(spec.transport.kind, "inprocess");
    assert_eq!(spec.storage.kind, "memory");
    assert!(!spec.workspace.enabled);
    // Unknown fields are tolerated, never folded into `metadata`.
    assert!(spec.metadata.is_empty());

    // Every omitted knob keeps its Python default.
    assert_eq!(spec.lifecycle, TeamLifecycle::Temporary);
    assert_eq!(spec.conversation_mode, ConversationMode::Team);
    assert_eq!(spec.dispatch_mode, DispatchMode::Autonomous);
    assert_eq!(spec.spawn_mode, SpawnMode::InProcess);
    assert_eq!(spec.bootstrap_mode, BootstrapMode::Leader);
    assert_eq!(spec.teammate_mode, MemberMode::BuildMode);
    assert_eq!(spec.language, None);
    assert_eq!(spec.resolved_language(), "cn");
    assert_eq!(spec.default_max_review_rounds, 3);
    assert_eq!(spec.review_stall_timeout, 2400);
    assert_eq!(spec.stale_claim_idle_timeout, 600);
    assert_eq!(spec.stale_pending_idle_timeout, 600);
    assert_eq!(spec.steer_batch_size, 2);
    assert!(!spec.enable_member_messaging);
    assert!(!spec.enable_task_verification);
    assert_eq!(spec.max_subtask_depth, 0);
}

#[test]
fn full_spec_parses_from_a_json_string() {
    let text = serde_json::to_string(&full_spec_json()).expect("the fixture serializes");
    let spec: TeamAgentSpec = serde_json::from_str(&text).expect("the full blueprint must parse");
    assert_eq!(spec.team_name, "beta");
    assert_eq!(spec.leader_member_name(), "team_leader");
}

#[test]
fn validate_rejects_a_missing_leader_blueprint() {
    let spec = TeamAgentSpec::default();
    let error = spec
        .validate()
        .expect_err("a spec without agents must not validate");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error
            .to_string()
            .contains("agents dict must contain a 'leader' key"),
        "{error}"
    );
}

#[test]
fn validate_rejects_reserved_predefined_member_names() {
    for name in ["user", "team_leader"] {
        let error = spec_with_predefined(name)
            .validate()
            .expect_err("a reserved predefined member must not validate");
        assert_eq!(error.kind(), ErrorKind::Validation, "member '{name}'");
        let text = error.to_string();
        assert!(text.contains("reserved name"), "member '{name}': {text}");
        assert!(text.contains(name), "member '{name}': {text}");
    }
}

#[test]
fn validate_rejects_framework_bootstrap_outside_scheduled_dispatch() {
    let spec = TeamAgentSpec {
        bootstrap_mode: BootstrapMode::Framework,
        dispatch_mode: DispatchMode::Autonomous,
        ..sample_spec()
    };
    let error = spec
        .validate()
        .expect_err("framework bootstrap requires scheduled dispatch");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let text = error.to_string();
    assert!(
        text.contains("bootstrap_mode='framework' requires dispatch_mode='scheduled'"),
        "{text}"
    );

    // The same blueprint validates once the dispatch mode agrees.
    let scheduled = TeamAgentSpec {
        bootstrap_mode: BootstrapMode::Framework,
        dispatch_mode: DispatchMode::Scheduled,
        ..sample_spec()
    };
    scheduled.validate().expect("scheduled framework browsing");
}

#[test]
fn sample_spec_validates_and_differs_only_in_verification() {
    sample_spec()
        .validate()
        .expect("the sample blueprint validates");
    let verified = sample_spec_verified();
    verified
        .validate()
        .expect("the verified blueprint validates");
    assert!(!sample_spec().enable_task_verification);
    assert!(verified.enable_task_verification);
    assert_eq!(verified.team_name, "alpha");
}

#[tokio::test]
async fn build_wires_the_leader_and_the_roster_once() {
    let factory = FakeHarnessFactory::new();
    let workspace = FakeWorkspaceProvider::new();
    let store: Arc<InMemoryTeamStore> = Arc::new(InMemoryTeamStore::new());

    let spec = sample_spec();
    let agent = spec
        .build_with(build_deps(&factory, &workspace, Some(Arc::clone(&store))))
        .await
        .expect("the sample blueprint builds");

    // Identity: the card is `<team>_<leader>` and the handle is the leader.
    assert_eq!(agent.card().name, "alpha_team_leader");
    assert_eq!(agent.role(), TeamRole::Leader);
    assert_eq!(agent.leader_member_name(), "team_leader");
    assert_eq!(agent.team_name(), "alpha");
    assert_eq!(agent.member_name(), "team_leader");
    assert_eq!(agent.language(), "cn");

    // Roster: sorted, leader plus the predefined member and the shared teammate blueprint.
    assert_eq!(
        agent.roster(),
        vec![
            "dev-1".to_owned(),
            "team_leader".to_owned(),
            "teammate".to_owned()
        ]
    );

    // The leader row carries its blueprint prompt; the teammate rows are cold (`unstarted`, no
    // runtime attached) because a teammate is only materialized when the leader spawns it.
    let leader = agent
        .backend()
        .member("team_leader")
        .expect("the leader has a row");
    assert_eq!(leader.status(), MemberStatus::Unstarted);
    assert_eq!(leader.role(), TeamRole::Leader);
    assert_eq!(leader.record().prompt, "lead the team");
    assert!(leader.runtime().is_some(), "the leader has a runtime");

    let dev = agent.backend().member("dev-1").expect("dev-1 has a row");
    assert_eq!(dev.status(), MemberStatus::Unstarted);
    assert_eq!(dev.role(), TeamRole::Teammate);
    assert_eq!(dev.record().prompt, "you are dev-1");
    assert!(dev.runtime().is_none(), "a cold teammate has no runtime");

    // The factory was asked for the leader only, with the leader's own workspace.
    let calls = factory.calls();
    assert_eq!(calls.len(), 1, "only the leader is built eagerly");
    assert_eq!(calls[0].member_name, "team_leader");
    assert_eq!(calls[0].team_name, "alpha");
    let allocated = workspace
        .root_for("team_leader")
        .expect("the leader workspace was allocated");
    assert_eq!(calls[0].workspace_root, allocated);
    assert_eq!(factory.created_members(), vec!["team_leader".to_owned()]);
    assert_eq!(workspace.allocations(), vec!["team_leader".to_owned()]);

    // The roster is persisted, one row per member.
    assert_eq!(store.list_members("alpha").await.expect("roster").len(), 3);
}

#[tokio::test]
async fn language_resolution_prefers_deps_then_spec_and_never_mutates() {
    let authored = TeamAgentSpec {
        language: Some("cf".to_owned()),
        ..sample_spec()
    };

    // A declared dependency language wins over the spec's own.
    let factory = FakeHarnessFactory::new();
    let workspace = FakeWorkspaceProvider::new();
    let agent = authored
        .clone()
        .build_with(build_deps(&factory, &workspace, None).with_language("en"))
        .await
        .expect("the blueprint builds with a declared language");
    assert_eq!(agent.spec().language.as_deref(), Some("en"));
    assert_eq!(agent.language(), "en");
    assert_eq!(factory.calls()[0].language, "en");
    assert_eq!(
        authored.language.as_deref(),
        Some("cf"),
        "build_with must not mutate the authored spec (documented deviation from Python)"
    );

    // Without one, the spec's own language wins.
    let factory = FakeHarnessFactory::new();
    let workspace = FakeWorkspaceProvider::new();
    let agent = authored
        .clone()
        .build_with(build_deps(&factory, &workspace, None))
        .await
        .expect("the blueprint builds with its own language");
    assert_eq!(agent.spec().language.as_deref(), Some("cf"));
    assert_eq!(agent.language(), "cf");

    // A spec that declares nothing lands on the `cn` default.
    let factory = FakeHarnessFactory::new();
    let workspace = FakeWorkspaceProvider::new();
    let spec = sample_spec();
    let agent = spec
        .build_with(build_deps(&factory, &workspace, None))
        .await
        .expect("the blueprint builds with the default language");
    assert_eq!(agent.spec().language.as_deref(), Some("cn"));
    assert_eq!(agent.language(), "cn");
    assert_eq!(spec.language, None, "the authored spec keeps its `None`");
}

#[tokio::test]
async fn activation_creates_then_attaches_in_the_same_session() {
    let fixture = Fixture::new();
    let spec = sample_spec();

    let first = fixture
        .manager
        .activate(&spec, "session-a", &Value::Null)
        .await
        .expect("the first activation creates the team");
    assert_eq!(first.team_name(), "alpha");
    assert!(fixture.manager.live("alpha").is_some());
    assert_eq!(
        fixture
            .manager
            .live_state("alpha")
            .map(|state| state.session_id),
        Some("session-a".to_owned())
    );

    let second = fixture
        .manager
        .activate(&spec, "session-a", &Value::Null)
        .await
        .expect("a second activation in the same session attaches");
    assert_eq!(second.team_name(), "alpha");
    assert!(
        Arc::ptr_eq(&first, &second),
        "attaching must hand back the live handle, not build a second team"
    );
    assert!(fixture.manager.live("alpha").is_some());
    assert_eq!(
        fixture.factory.created_members(),
        vec!["team_leader".to_owned()],
        "the team is built exactly once"
    );
}

#[tokio::test]
async fn activation_rejects_a_temporary_team_from_another_session() {
    let fixture = Fixture::new();
    let spec = sample_spec();
    fixture
        .manager
        .activate(&spec, "session-a", &Value::Null)
        .await
        .expect("the first activation creates the team");

    let error = fixture
        .manager
        .activate(&spec, "session-b", &Value::Null)
        .await
        .expect_err("a temporary team must not serve a second session");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let text = error.to_string();
    assert!(
        text.contains("cannot serve session 'session-b'"),
        "the rejection names the refused session: {text}"
    );
    assert!(
        text.contains("session-a"),
        "the rejection names the owning session: {text}"
    );
    assert!(
        fixture.manager.live("alpha").is_some(),
        "a rejected request must not tear the live team down"
    );
}

#[tokio::test]
async fn activation_resumes_a_persistent_team_in_another_session() {
    let fixture = Fixture::new();
    let spec = TeamAgentSpec {
        lifecycle: TeamLifecycle::Persistent,
        ..sample_spec()
    };

    fixture
        .manager
        .activate(&spec, "session-a", &Value::Null)
        .await
        .expect("the first activation creates the persistent team");

    let resumed = fixture
        .manager
        .activate(&spec, "session-b", &Value::Null)
        .await
        .expect("a persistent team resumes for a second session");
    assert_eq!(resumed.team_name(), "alpha");
    assert!(fixture.manager.live("alpha").is_some());
    assert_eq!(
        fixture.factory.created_members(),
        vec!["team_leader".to_owned()],
        "resuming must not build a second team"
    );
}

#[tokio::test]
async fn activation_rejects_an_invalid_blueprint() {
    let fixture = Fixture::new();
    let error = fixture
        .manager
        .activate(&TeamAgentSpec::default(), "session-a", &Value::Null)
        .await
        .expect_err("an invalid blueprint must not activate");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error
            .to_string()
            .contains("agents dict must contain a 'leader' key"),
        "{error}"
    );
    assert!(fixture.manager.live("agent_team").is_none());
}
