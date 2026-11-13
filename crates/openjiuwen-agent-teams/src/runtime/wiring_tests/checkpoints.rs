//! Per-member checkpoint wiring: team store, member identity and the capture path.
//!
//! The seam under test is `TeamAgentSpec::metadata["checkpoint"]` plus the identity the runtime
//! gives each member runtime:
//!
//! | fact | effect |
//! | --- | --- |
//! | `build_with` | the team store is resolved once (injected wins) and pinned in the dependencies |
//! | no injection | the default build pins one `"memory"` team store (the file backend pins none) |
//! | leader / teammate build | the runtime is keyed on `(derived member session, card id)` and cwd = workspace root |
//! | `start` | a configured and enabled runtime gets a checkpoint sink and the policy's resume policy |
//! | a finished turn / `stop` | the store receives one record for that identity |
//! | a restarted member | that record's provider envelope is restored as the start context's checkpoint |
//! | `{"enabled": false}` | no store, no sink, no capture (leader and teammate alike) |
//!
//! Identity values: a member runs under `"<team session>:<member>"` — both the runtime session id
//! and the harness `HarnessContext::host_session_id`, because `start(None)` makes the derived id the
//! host session — and its checkpoints are keyed on the blueprint card id, falling back to the member
//! name when the blueprint carries no card.
//!
//! Every test is offline and deterministic: scripted harnesses, a recording (or typed in-memory)
//! checkpoint store, the recording workspace provider, temp directories and [`BOUND`]-limited waits.

use std::sync::Arc;

use openjiuwen_common::utils::id::uuid_hex;
use openjiuwen_harness_protocol::ResumePolicy;
use serde_json::json;

use crate::harness::{
    FileMemberCheckpointStore, InMemoryMemberCheckpointStore, MemberCheckpointStore,
};
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

#[tokio::test]
async fn a_default_build_pins_one_memory_team_store() {
    let wiring = wire(&blueprint("alpha", "boss", &["dev-1"]), |deps| deps).await;
    let store = wiring
        .agent
        .spawner()
        .deps()
        .checkpoints
        .as_ref()
        .expect("an enabled default policy pins a team store")
        .clone();
    assert_eq!(
        store.name(),
        "memory",
        "the default backend is one in-memory store shared by the whole team"
    );

    // The file backend has no team-wide root: every member resolves its own directory.
    let file = with_metadata(
        blueprint("alpha", "boss", &["dev-1"]),
        CHECKPOINT_METADATA_KEY,
        json!({"store": "file"}),
    );
    let wiring = wire(&file, |deps| deps).await;
    assert!(
        wiring.agent.spawner().deps().checkpoints.is_none(),
        "a file config pins no team store"
    );
}

#[tokio::test]
async fn an_injected_store_reaches_the_leader_and_a_teammate() {
    let (recorder, store) = store_and_recorder();
    let spec = blueprint("alpha", "boss", &["dev-1"]);
    let leader_card = spec
        .agents
        .get("leader")
        .and_then(|spec| spec.card.as_ref())
        .expect("the leader blueprint carries a card")
        .id
        .clone();
    let member_card = spec
        .agents
        .get("dev-1")
        .and_then(|spec| spec.card.as_ref())
        .expect("dev-1 has a card")
        .id
        .clone();
    let wiring = wire(&spec, move |deps| deps.with_checkpoint_store(store)).await;
    let session = wiring.agent.session_id().to_owned();

    wiring.agent.start().await.unwrap();
    wiring.agent.spawn_member("dev-1").await.unwrap();

    assert_eq!(
        recorder.loads(),
        vec![
            (format!("{session}:boss"), leader_card),
            (format!("{session}:dev-1"), member_card),
        ],
        "both members restore under the derived session id and the blueprint card id"
    );
}

#[tokio::test]
async fn a_leader_without_a_card_restores_under_its_member_name() {
    let (recorder, store) = store_and_recorder();
    let mut spec = blueprint("alpha", "boss", &[]);
    spec.agents
        .get_mut("leader")
        .expect("the leader has a blueprint")
        .card = None;
    let wiring = wire(&spec, move |deps| deps.with_checkpoint_store(store)).await;
    let session = wiring.agent.session_id().to_owned();

    wiring.agent.start().await.unwrap();

    assert_eq!(
        recorder.loads(),
        vec![(format!("{session}:boss"), "boss".to_owned())],
        "a card-less leader blueprint keys the checkpoint on the leader member name"
    );
}

#[tokio::test]
async fn a_disabled_policy_attaches_nothing_to_a_teammate_either() {
    let (recorder, store) = store_and_recorder();
    let spec = with_metadata(
        blueprint("alpha", "boss", &["dev-1"]),
        CHECKPOINT_METADATA_KEY,
        json!({"enabled": false}),
    );
    let wiring = wire(&spec, move |deps| deps.with_checkpoint_store(store)).await;

    wiring.agent.start().await.unwrap();
    wiring.agent.spawn_member("dev-1").await.unwrap();
    wiring
        .agent
        .shutdown_member("dev-1", MemberLifetime::Shutdown, false)
        .await
        .unwrap();

    let harness = wiring.factory.harness("dev-1").expect("dev-1 harness");
    let contexts = harness.contexts();
    let context = contexts.first().expect("dev-1 started once");
    assert!(
        context.checkpoint_sink.is_none(),
        "a disabled policy configures no sink for a teammate"
    );
    assert!(recorder.loads().is_empty(), "nothing is restored");
    assert_eq!(recorder.saves(), 0, "nothing is captured, not even on stop");
}

#[tokio::test]
async fn a_restarted_member_restores_the_stored_record() {
    // The typed store is the same team-shared route the default in-memory backend takes, but it can
    // be inspected, so the restore is asserted instead of inferred.
    let typed = Arc::new(InMemoryMemberCheckpointStore::new());
    let factory = ScriptedFactory::new();
    factory.with_turn_events();
    let workspace = RecordingWorkspaceProvider::new();
    let spec = blueprint("alpha", "boss", &["dev-1"]);
    let member_card = spec
        .agents
        .get("dev-1")
        .and_then(|spec| spec.card.as_ref())
        .expect("dev-1 has a card")
        .id
        .clone();
    let agent = spec
        .build_with(
            deps(&factory, &workspace)
                .with_checkpoint_store(Arc::clone(&typed) as Arc<dyn MemberCheckpointStore>),
        )
        .await
        .expect("the team did not build");
    let session = agent.session_id().to_owned();
    let member_session = format!("{session}:dev-1");

    agent.start().await.unwrap();
    agent.spawn_member("dev-1").await.unwrap();
    agent
        .backend()
        .member("dev-1")
        .and_then(|member| member.runtime())
        .expect("dev-1 has a live runtime")
        .send("work".to_owned(), false)
        .await
        .unwrap();

    wait_until("the finished turn's capture", || {
        typed.get(&member_session, &member_card).is_some()
    })
    .await;
    assert_eq!(typed.len(), 1, "one member, one stored identity");
    assert!(
        typed.get(&member_session, &member_card).is_some(),
        "the record is keyed on the derived session and the card id"
    );

    agent
        .shutdown_member("dev-1", MemberLifetime::Shutdown, false)
        .await
        .unwrap();
    // A retired row has no `shut_down -> unstarted` edge, so a restart drops the live row and lets
    // `spawn_member` materialize a fresh unstarted one (the store row is overwritten by the upsert).
    assert!(
        agent.backend().remove_member("dev-1").is_some(),
        "the retired member was on the live roster"
    );
    agent.spawn_member("dev-1").await.unwrap();

    let harness = factory.harness("dev-1").expect("the restarted harness");
    let contexts = harness.contexts();
    assert_eq!(contexts.len(), 1, "the restarted harness started once");
    let context = &contexts[0];
    let restored = context
        .checkpoint
        .as_ref()
        .expect("the restarted member resumes from the stored envelope");
    assert_eq!(
        restored.provider, "scripted",
        "the provider envelope came back"
    );
    assert_eq!(context.resume_policy, ResumePolicy::ResumeIfAvailable);
    assert_eq!(
        context.host_session_id, member_session,
        "start(None) makes the derived member session the host session"
    );
    assert_eq!(
        typed.len(),
        1,
        "the restart read the store instead of minting a second identity"
    );
}

#[tokio::test]
async fn a_named_file_dir_receives_one_checkpoint_document() {
    let directory = std::env::temp_dir().join(format!("openjiuwen-cp-{}", uuid_hex()));
    let factory = ScriptedFactory::new();
    factory.with_turn_events();
    let workspace = RecordingWorkspaceProvider::new();
    let mut spec = with_metadata(
        blueprint("alpha", "boss", &["dev-1"]),
        CHECKPOINT_METADATA_KEY,
        json!({"store": "file", "dir": directory.to_string_lossy()}),
    );
    spec.agents
        .get_mut("dev-1")
        .expect("dev-1 has a blueprint")
        .card = None;
    let agent = spec
        .build_with(deps(&factory, &workspace))
        .await
        .expect("the team did not build");
    let session = agent.session_id().to_owned();

    agent.start().await.unwrap();
    agent.spawn_member("dev-1").await.unwrap();
    agent
        .backend()
        .member("dev-1")
        .and_then(|member| member.runtime())
        .expect("dev-1 has a live runtime")
        .send("work".to_owned(), false)
        .await
        .unwrap();

    let store = FileMemberCheckpointStore::new(&directory);
    let path = store.path_for(&format!("{session}:dev-1"), "dev-1");
    wait_until("the member checkpoint document", || path.exists()).await;

    let text = std::fs::read_to_string(&path).expect("the checkpoint document is readable");
    let value: serde_json::Value =
        serde_json::from_str(&text).expect("the checkpoint document is JSON");
    assert_eq!(
        value["checkpoint"]["session_id"],
        json!(format!("{session}:dev-1")),
        "the document carries the derived member session"
    );
    assert_eq!(value["checkpoint"]["card_id"], json!("dev-1"));
    assert_eq!(
        value["protocol"]["provider"],
        json!("scripted"),
        "the provider envelope is stored beside the portable payload"
    );

    agent
        .shutdown_member("dev-1", MemberLifetime::Shutdown, false)
        .await
        .unwrap();
    std::fs::remove_dir_all(&directory).expect("the temp checkpoint dir is removed");
}
