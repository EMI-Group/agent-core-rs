#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Member-scoped checkpointing across one shared checkpointer.
//!
//! The headline scenario is two members of one team sharing a single
//! `MemoryCheckpointer`: each saves different state, and after a second team
//! session object with the same session id and fresh children restores, every
//! member sees exactly its own state. Persisting per member is the Python
//! parity requirement of `AgentStorage` in
//! `openjiuwen/core/session/checkpointer/inmemory.py`, whose agent store is
//! keyed by `session.agent_id()`.
//!
//! Everything here is in-process: a memory checkpointer, real
//! `AgentTeamSession`/`AgentSession` children, and no model or network.
use std::sync::Arc;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_multiagent::member_checkpoint::{
    member_checkpoint_id, member_checkpoint_key, MemberCheckpointer, MEMBER_CHECKPOINT_NAMESPACE,
};
use openjiuwen_session::{
    AgentSession, AgentTeamSession, AgentTeamSessionOptions, CheckpointNamespace, Checkpointer,
    MemoryCheckpointer, Session,
};
use serde_json::json;

/// The team session id shared by the saved and the restored run.
const TEAM_SESSION: &str = "team-session-1";

/// Creates a team session with a fixed id over `checkpointer`.
fn team_session(checkpointer: Arc<dyn Checkpointer>) -> Arc<AgentTeamSession> {
    Arc::new(AgentTeamSession::with_options(AgentTeamSessionOptions {
        session_id: Some(TEAM_SESSION.to_string()),
        team_id: Some("crew".to_string()),
        checkpointer: Some(checkpointer),
        ..AgentTeamSessionOptions::default()
    }))
}

/// A checkpointer handle plus the concrete store behind it.
fn shared_store() -> (MemoryCheckpointer, Arc<dyn Checkpointer>) {
    let store = MemoryCheckpointer::new();
    let shared: Arc<dyn Checkpointer> = Arc::new(store.clone());
    (store, shared)
}

#[tokio::test]
async fn two_members_restore_only_their_own_state() {
    let (store, shared) = shared_store();
    let checkpointer = MemberCheckpointer::new(Arc::clone(&shared));

    let team = team_session(Arc::clone(&shared));
    let researcher = team.create_agent_session("researcher");
    researcher
        .update_state(json!({ "notes": ["r1", "r2"] }))
        .await
        .unwrap();
    researcher
        .update_component_state("memory", json!({ "hits": 3 }))
        .await
        .unwrap();
    let writer = team.create_agent_session("writer");
    writer.update_state(json!({ "draft": "w1" })).await.unwrap();

    let member_sessions: Vec<String> = vec![researcher.id().to_string(), writer.id().to_string()];
    let members: Vec<Arc<dyn Session>> = vec![Arc::new(researcher), Arc::new(writer)];
    let checkpoint_ids = checkpointer.save_all(TEAM_SESSION, &members).await.unwrap();
    assert_eq!(checkpoint_ids.len(), 2);

    // Both member keys exist under the team session, one per member, with
    // distinct entity ids and the agent namespace.
    for ((checkpoint_id, agent_id), member_session_id) in checkpoint_ids
        .iter()
        .zip(["researcher", "writer"])
        .zip(&member_sessions)
    {
        let key = member_checkpoint_key(TEAM_SESSION, agent_id);
        assert!(
            checkpoint_id.starts_with(&format!("{key}:")),
            "checkpoint id {checkpoint_id} is not under {key}"
        );
        let saved = store
            .load(TEAM_SESSION, checkpoint_id)
            .await
            .unwrap()
            .expect("member checkpoint was saved");
        assert_eq!(saved.session_id, TEAM_SESSION);
        assert_eq!(saved.namespace, MEMBER_CHECKPOINT_NAMESPACE);
        assert_eq!(saved.namespace, CheckpointNamespace::Agent);
        assert_eq!(saved.entity_id, agent_id);
        assert_eq!(saved.metadata["agent_session_id"], json!(member_session_id));
    }
    assert_eq!(store.len(TEAM_SESSION), 2);
    assert_eq!(
        checkpointer.saved_members(TEAM_SESSION).await.unwrap(),
        vec!["researcher".to_string(), "writer".to_string()]
    );

    // A second team session object with the same session id and fresh children.
    let replay = team_session(Arc::clone(&shared));
    let fresh_researcher = replay.create_agent_session("researcher");
    let fresh_writer = replay.create_agent_session("writer");
    let restored = checkpointer
        .restore_all(TEAM_SESSION, &[&fresh_researcher, &fresh_writer])
        .await
        .unwrap();
    assert_eq!(
        restored,
        vec!["researcher".to_string(), "writer".to_string()]
    );

    assert_eq!(
        fresh_researcher.get_state("").await.unwrap(),
        Some(json!({ "notes": ["r1", "r2"] }))
    );
    assert_eq!(
        fresh_researcher
            .get_component_state("memory")
            .await
            .unwrap(),
        Some(json!({ "hits": 3 }))
    );
    assert_eq!(
        fresh_writer.get_state("").await.unwrap(),
        Some(json!({ "draft": "w1" }))
    );
    // Neither member sees the other's state.
    assert_eq!(fresh_researcher.get_state("draft").await.unwrap(), None);
    assert_eq!(fresh_writer.get_state("notes").await.unwrap(), None);
    assert_eq!(
        fresh_writer.get_component_state("memory").await.unwrap(),
        None
    );
}

#[tokio::test]
async fn save_without_a_team_parent_saves_nothing() {
    let (store, shared) = shared_store();
    let checkpointer = MemberCheckpointer::new(shared);
    let solo = AgentSession::new("standalone-agent");

    assert_eq!(checkpointer.save(&solo).await.unwrap(), None);
    assert!(!checkpointer.restore(&solo).await.unwrap());
    assert_eq!(store.len(TEAM_SESSION), 0);
}

#[tokio::test]
async fn the_latest_save_of_a_member_wins() {
    let (_store, shared) = shared_store();
    let checkpointer = MemberCheckpointer::new(Arc::clone(&shared));
    let team = team_session(shared);
    let member = team.create_agent_session("researcher");

    member.update_state(json!({ "step": 1 })).await.unwrap();
    let first = checkpointer.save(&member).await.unwrap().unwrap();
    member.update_state(json!({ "step": 2 })).await.unwrap();
    let second = checkpointer.save(&member).await.unwrap().unwrap();
    assert_eq!(first, member_checkpoint_id(TEAM_SESSION, "researcher", 0));
    assert_eq!(second, member_checkpoint_id(TEAM_SESSION, "researcher", 1));

    let loaded = checkpointer
        .load(TEAM_SESSION, "researcher")
        .await
        .unwrap()
        .expect("member checkpoint exists");
    assert_eq!(loaded.checkpoint_id, second);
    assert_eq!(loaded.global_state, json!({ "step": 2 }));

    drop(team);
    let fresh_team = team_session(checkpointer.checkpointer());
    let fresh = fresh_team.create_agent_session("researcher");
    assert!(checkpointer.restore(&fresh).await.unwrap());
    assert_eq!(fresh.get_state("step").await.unwrap(), Some(json!(2)));
}

#[tokio::test]
async fn saved_members_is_sorted_and_de_duplicated() {
    let (_store, shared) = shared_store();
    let checkpointer = MemberCheckpointer::new(shared);
    let team = team_session(checkpointer.checkpointer());
    for agent_id in ["writer", "researcher", "researcher"] {
        let member = team.create_agent_session(agent_id);
        checkpointer.save(&member).await.unwrap();
    }

    assert_eq!(
        checkpointer.saved_members(TEAM_SESSION).await.unwrap(),
        vec!["researcher".to_string(), "writer".to_string()]
    );
    assert_eq!(
        checkpointer.saved_members("no-such-team").await.unwrap(),
        Vec::<String>::new()
    );
}

#[tokio::test]
async fn restoring_an_unknown_member_reports_false() {
    let (_store, shared) = shared_store();
    let checkpointer = MemberCheckpointer::new(shared);
    let team = team_session(checkpointer.checkpointer());
    let ghost = team.create_agent_session("ghost");

    assert!(!checkpointer.restore(&ghost).await.unwrap());
    assert_eq!(
        checkpointer
            .restore_all(TEAM_SESSION, &[&ghost])
            .await
            .unwrap(),
        Vec::<String>::new()
    );
}

#[tokio::test]
async fn an_empty_agent_id_is_a_validation_error() {
    let (_store, shared) = shared_store();
    let checkpointer = MemberCheckpointer::new(shared);
    let team = team_session(checkpointer.checkpointer());
    let member = team.create_agent_session("researcher");

    let error = checkpointer
        .save_in(TEAM_SESSION, "", &member)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("agent id"));

    let error = checkpointer.load(TEAM_SESSION, "").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[tokio::test]
async fn member_checkpoints_do_not_collide_with_the_team_checkpoint() {
    let (store, shared) = shared_store();
    let checkpointer = MemberCheckpointer::new(Arc::clone(&shared));
    let team = team_session(Arc::clone(&shared));
    team.update_state(json!({ "phase": "running" }))
        .await
        .unwrap();
    // The team's own checkpoint shares the checkpointer and the session id.
    let team_checkpoint_id = team.commit().await.unwrap().expect("team checkpoint saved");

    let member = team.create_agent_session("researcher");
    member
        .update_state(json!({ "phase": "member" }))
        .await
        .unwrap();
    checkpointer.save(&member).await.unwrap().unwrap();

    let member_state = checkpointer
        .load(TEAM_SESSION, "researcher")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(member_state.global_state, json!({ "phase": "member" }));

    let team_state = store
        .load(TEAM_SESSION, &team_checkpoint_id)
        .await
        .unwrap()
        .expect("team checkpoint kept");
    assert_eq!(team_state.namespace, CheckpointNamespace::AgentTeam);
    assert_eq!(team_state.entity_id, "crew");
    assert_eq!(team_state.global_state, json!({ "phase": "running" }));

    assert_eq!(store.len(TEAM_SESSION), 2);
    assert_eq!(
        checkpointer.saved_members(TEAM_SESSION).await.unwrap(),
        vec!["researcher".to_string()]
    );
}
