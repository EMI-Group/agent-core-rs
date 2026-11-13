//! Per-member checkpoint wiring: team store, member identity and the capture path.
//!
//! The seam under test is `TeamAgentSpec::metadata["checkpoint"]` plus the identity the runtime
//! gives each member runtime:
//!
//! | fact | effect |
//! | --- | --- |
//! | `build_with` | the team store is resolved once (injected wins) and pinned in the dependencies |
//! | leader / teammate build | the runtime is keyed on `(derived member session, card id)` and cwd = workspace root |
//! | `start` | a configured and enabled runtime gets a checkpoint sink and the policy's resume policy |
//! | a finished turn / `stop` | the store receives one record for that identity |
//! | `{"enabled": false}` | no store, no sink, no capture |
//!
//! Every test is offline and deterministic: scripted harnesses, a recording checkpoint store, the
//! recording workspace provider and [`BOUND`]-limited waits.

use std::sync::Arc;

use openjiuwen_harness_protocol::ResumePolicy;
use serde_json::json;

use crate::harness::{FileMemberCheckpointStore, MemberCheckpointStore};
use crate::runtime::checkpoints::CHECKPOINT_DIR;
use crate::runtime::spec_meta::CHECKPOINT_METADATA_KEY;
use crate::runtime::team_agent::test_support::RecordingCheckpointStore;
use crate::runtime::team_agent::SpecBuild;
use crate::schema::MemberLifetime;

use super::{
    blueprint, deps, wait_until, wire, with_metadata, RecordingWorkspaceProvider, ScriptedFactory,
};

/// The injected store as a trait object, keeping the recorder for its counters.
fn store_and_recorder() -> (
    Arc<RecordingCheckpointStore>,
    Arc<dyn MemberCheckpointStore>,
) {
    let recorder = Arc::new(RecordingCheckpointStore::default());
    let store: Arc<dyn MemberCheckpointStore> =
        Arc::clone(&recorder) as Arc<dyn MemberCheckpointStore>;
    (recorder, store)
}

#[tokio::test]
async fn the_build_hands_the_leader_a_checkpoint_sink() {
    let store = RecordingCheckpointStore::shared();
    let wiring = wire(&blueprint("alpha", "boss", &["dev-1"]), |deps| {
        deps.with_checkpoint_store(Arc::clone(&store))
    })
    .await;
    wiring.agent.start().await.unwrap();

    let harness = wiring.factory.harness("boss").expect("the leader harness");
    let contexts = harness.contexts();
    assert_eq!(contexts.len(), 1, "the leader started once");
    let context = &contexts[0];
    assert!(
        context.checkpoint_sink.is_some(),
        "a configured store attaches a sink"
    );
    assert_eq!(context.resume_policy, ResumePolicy::ResumeIfAvailable);
    let cwd = context
        .cwd
        .as_deref()
        .expect("the workspace root is the cwd");
    assert!(cwd.ends_with("boss"), "{cwd}");
    assert_eq!(store.name(), "recording");
}

#[tokio::test]
async fn a_finished_turn_is_captured_into_the_team_store() {
    let (recorder, store) = store_and_recorder();
    // The flag applies to harnesses built *after* it is set, so it precedes the build: a turn
    // capture needs the harness to close its turn with a terminal `turn_lifecycle` event.
    let factory = ScriptedFactory::new();
    factory.with_turn_events();
    let workspace = RecordingWorkspaceProvider::new();
    let agent = blueprint("alpha", "boss", &[])
        .build_with(deps(&factory, &workspace).with_checkpoint_store(store))
        .await
        .expect("the team did not build");
    agent.start().await.unwrap();

    agent
        .runtime()
        .send("hello".to_owned(), false)
        .await
        .unwrap();

    wait_until("the turn checkpoint", || recorder.saves() >= 1).await;
    assert_eq!(recorder.saves(), 1, "one finished turn, one capture");
}

#[tokio::test]
async fn a_member_restores_under_its_derived_session_and_card_identity() {
    let (recorder, store) = store_and_recorder();
    let mut spec = blueprint("alpha", "boss", &["dev-1"]);
    spec.agents
        .get_mut("dev-1")
        .expect("dev-1 has a blueprint")
        .card = None;
    let wiring = wire(&spec, move |deps| deps.with_checkpoint_store(store)).await;
    let session = wiring.agent.session_id().to_owned();

    wiring.agent.spawn_member("dev-1").await.unwrap();

    assert_eq!(
        recorder.loads(),
        vec![(format!("{session}:dev-1"), "dev-1".to_owned())],
        "a card-less blueprint keys the checkpoint on the member name"
    );
}

#[tokio::test]
async fn a_disabled_policy_attaches_no_sink() {
    let spec = with_metadata(
        blueprint("alpha", "boss", &["dev-1"]),
        CHECKPOINT_METADATA_KEY,
        json!({"enabled": false}),
    );
    let wiring = wire(&spec, |deps| deps).await;
    wiring.agent.start().await.unwrap();

    let harness = wiring.factory.harness("boss").expect("the leader harness");
    let contexts = harness.contexts();
    let context = contexts.first().expect("the leader started once");
    assert!(
        context.checkpoint_sink.is_none(),
        "a disabled policy configures no sink"
    );
    assert_eq!(context.resume_policy, ResumePolicy::New);
}

#[tokio::test]
async fn a_file_config_roots_each_member_under_its_own_workspace() {
    let mut spec = with_metadata(
        blueprint("alpha", "boss", &["dev-1"]),
        CHECKPOINT_METADATA_KEY,
        json!({"store": "file"}),
    );
    spec.agents
        .get_mut("dev-1")
        .expect("dev-1 has a blueprint")
        .card = None;
    let wiring = wire(&spec, |deps| deps).await;
    let session = wiring.agent.session_id().to_owned();

    wiring.agent.spawn_member("dev-1").await.unwrap();
    wiring
        .agent
        .shutdown_member("dev-1", MemberLifetime::Shutdown, false)
        .await
        .unwrap();

    // The recording workspace provider roots a member at `<temp>/…/<team>/<session>/<member>`.
    let workspace_root = std::env::temp_dir()
        .join("openjiuwen-agent-teams-tests")
        .join("alpha")
        .join(&session)
        .join("dev-1");
    let expected = FileMemberCheckpointStore::new(workspace_root.join(CHECKPOINT_DIR))
        .path_for(&format!("{session}:dev-1"), "dev-1");
    assert!(
        expected.exists(),
        "the stop capture wrote {expected:?} under the member workspace"
    );
}
