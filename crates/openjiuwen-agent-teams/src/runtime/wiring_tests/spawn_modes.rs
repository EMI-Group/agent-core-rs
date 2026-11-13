//! `metadata["spawn"]` → the child spawner that materialises a member, and the in-process default.
//!
//! The spawn mode is parsed from the blueprint (`metadata["spawn"]`, else `spawn_mode`) and travels
//! to [`TeamMemberSpawner`](crate::runtime::spawner::TeamMemberSpawner) as
//! [`MemberSpawnOptions`](crate::member::MemberSpawnOptions). These tests pin which backend each
//! mode selects, that a non-in-process mode bypasses the harness factory entirely, and that a
//! failed launch leaves no backend registered.

use std::sync::Arc;

use serde_json::{json, Value};

use openjiuwen_common::error::{ErrorKind, Result};

use crate::member::{
    MemberRuntime, MemberSpawnKind, MemberSpawnOptions, MemberSpawnRequest, MemberSpawner,
};
use crate::runtime::spec_meta::{
    fork_directive, spawn_options, FORK_METADATA_KEY, SPAWN_METADATA_KEY,
};
use crate::runtime::team_agent::TeamAgent;
use crate::schema::{MemberStatus, SpawnMode, TeamAgentSpec};

use super::{blueprint, wire, with_metadata, ScriptedFactory};

/// One spawn attempt: the built team, its harness factory, and the spawner's outcome.
type SpawnOutcome = (
    Arc<TeamAgent>,
    Arc<ScriptedFactory>,
    Result<Arc<dyn MemberRuntime>>,
);

/// A shell command that reads stdin forever and never answers a member frame.
fn silent_reader() -> Value {
    json!({"program": "sh", "args": ["-c", "while read -r _; do :; done"]})
}

/// A bare spawn-options object selecting the OS-process backend with `command`.
fn process_metadata(command: &Value) -> Value {
    json!({"kind": "process", "process": {"command": command}})
}

/// A bare spawn-options object selecting the external-CLI backend with `command`.
fn external_cli_metadata(command: &Value) -> Value {
    json!({"kind": "external_cli", "external_cli": {"command": command}})
}

/// Builds a spawn request that carries `options`, the way the runtime hands them to a spawner.
fn request(member: &str, options: &MemberSpawnOptions) -> MemberSpawnRequest {
    MemberSpawnRequest::new("alpha", member, "session-1", std::env::temp_dir())
        .with_options(options.clone())
}

/// Builds the one-teammate blueprint every spawn test drives.
fn one_teammate() -> TeamAgentSpec {
    blueprint("alpha", "boss", &["dev-1"])
}

/// Builds `spec` and sends its spawn options through the spawner for `member`.
async fn spawn_one(spec: &TeamAgentSpec, member: &str) -> SpawnOutcome {
    let wiring = wire(spec, |deps| deps).await;
    let spawner = wiring.agent.spawner();
    let result = spawner.spawn(request(member, spawner.options())).await;
    (wiring.agent, wiring.factory, result)
}

#[tokio::test]
async fn the_default_blueprint_spawns_in_process() {
    let wiring = wire(&one_teammate(), |deps| deps).await;
    assert_eq!(
        wiring.agent.spawner().options().kind,
        MemberSpawnKind::InProcess
    );

    wiring
        .agent
        .spawn_member("dev-1")
        .await
        .expect("the in-process member did not spawn");

    assert_eq!(
        wiring.agent.spawner().spawn_kind("dev-1"),
        Some(MemberSpawnKind::InProcess)
    );
    assert!(
        !wiring.factory.calls().is_empty(),
        "the harness factory must build the member's brain"
    );
    assert!(
        wiring.factory.harness("dev-1").is_some(),
        "the harness of the spawned member was not built"
    );
}

#[tokio::test]
async fn process_metadata_spawns_and_reaps_a_child() {
    let spec = with_metadata(
        one_teammate(),
        SPAWN_METADATA_KEY,
        process_metadata(&silent_reader()),
    );
    let (agent, factory, result) = spawn_one(&spec, "dev-1").await;
    let spawner = agent.spawner();
    assert_eq!(spawner.options().kind, MemberSpawnKind::Process);
    let runtime = result.expect("the process member did not spawn");

    assert_eq!(spawner.spawn_kind("dev-1"), Some(MemberSpawnKind::Process));
    assert!(
        factory.harness("dev-1").is_none(),
        "a child-process member must not reach the harness factory"
    );

    spawner
        .despawn("dev-1")
        .await
        .expect("despawn did not reap the child");
    assert_eq!(spawner.spawn_kind("dev-1"), None);
    spawner
        .despawn("dev-1")
        .await
        .expect("a second despawn must stay a no-op");
    drop(runtime);
}

#[tokio::test]
async fn external_cli_metadata_spawns_and_reaps_a_child() {
    let spec = with_metadata(
        one_teammate(),
        SPAWN_METADATA_KEY,
        external_cli_metadata(&silent_reader()),
    );
    let (agent, factory, result) = spawn_one(&spec, "dev-1").await;
    let spawner = agent.spawner();
    assert_eq!(spawner.options().kind, MemberSpawnKind::ExternalCli);
    let runtime = result.expect("the external-CLI member did not spawn");

    assert_eq!(
        spawner.spawn_kind("dev-1"),
        Some(MemberSpawnKind::ExternalCli)
    );
    assert!(
        factory.harness("dev-1").is_none(),
        "an external-CLI member must not reach the harness factory"
    );

    spawner
        .despawn("dev-1")
        .await
        .expect("despawn did not reap the child");
    spawner
        .despawn("dev-1")
        .await
        .expect("a second despawn must stay a no-op");
    drop(runtime);
}

#[tokio::test]
async fn a_bare_process_mode_needs_the_member_binary() {
    let spec = with_metadata(
        one_teammate(),
        SPAWN_METADATA_KEY,
        json!({"kind": "process"}),
    );
    let (agent, _factory, result) = spawn_one(&spec, "dev-1").await;
    let error = result.err().expect("the default member command must fail");

    assert_eq!(error.kind(), ErrorKind::Config);
    let message = error.to_string();
    assert!(
        message.contains("was not found on PATH") || message.contains("is not an existing file"),
        "unexpected launch error: {message}"
    );
    assert_eq!(
        agent.spawner().spawn_kind("dev-1"),
        None,
        "a failed launch must not register a backend"
    );
}

#[tokio::test]
async fn a_bare_ssh_mode_needs_a_host() {
    let spec = with_metadata(one_teammate(), SPAWN_METADATA_KEY, json!({"kind": "ssh"}));
    let (agent, _factory, result) = spawn_one(&spec, "dev-1").await;
    let error = result.err().expect("an SSH spawn without a host must fail");

    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(
        error
            .to_string()
            .contains("ssh transport has no host configured"),
        "unexpected SSH error: {error}"
    );
    assert_eq!(agent.spawner().spawn_kind("dev-1"), None);
}

#[tokio::test]
async fn team_level_spawn_uses_the_metadata_selected_backend() {
    let spec = with_metadata(one_teammate(), SPAWN_METADATA_KEY, json!({"kind": "ssh"}));
    let wiring = wire(&spec, |deps| deps).await;
    assert_eq!(wiring.agent.spawner().options().kind, MemberSpawnKind::Ssh);

    let error = wiring
        .agent
        .spawn_member("dev-1")
        .await
        .expect_err("the SSH roster spawn must fail");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(error
        .to_string()
        .contains("ssh transport has no host configured"));
    assert!(
        wiring.factory.harness("dev-1").is_none(),
        "a failed spawn must not fall back to an in-process brain"
    );
    let row = wiring
        .agent
        .backend()
        .member("dev-1")
        .expect("the roster row dev-1");
    assert_eq!(row.status(), MemberStatus::Unstarted);

    let default = wire(&one_teammate(), |deps| deps).await;
    assert_eq!(
        default.agent.spawner().options().kind,
        MemberSpawnKind::InProcess
    );
}

#[test]
fn spawn_metadata_wins_over_the_typed_spawn_mode() {
    let mut spec = one_teammate();
    spec.spawn_mode = SpawnMode::Process;
    assert_eq!(spawn_options(&spec).kind, MemberSpawnKind::Process);

    let spec = with_metadata(spec, SPAWN_METADATA_KEY, json!({"kind": "external_cli"}));
    assert_eq!(
        spawn_options(&spec).kind,
        MemberSpawnKind::ExternalCli,
        "metadata must win over spawn_mode"
    );
}

#[test]
fn the_wrapper_parses_options_and_fork_together() {
    let spec = with_metadata(
        one_teammate(),
        SPAWN_METADATA_KEY,
        json!({"options": {"kind": "process"}, "fork": {"parent": "boss"}}),
    );
    assert_eq!(spawn_options(&spec).kind, MemberSpawnKind::Process);
    let fork = fork_directive(&spec).expect("the fork directive did not parse");
    assert_eq!(fork.parent, "boss");

    let spec = with_metadata(one_teammate(), FORK_METADATA_KEY, json!({"parent": "boss"}));
    let fork = fork_directive(&spec).expect("the standalone fork key did not parse");
    assert_eq!(fork.parent, "boss");
}

#[test]
fn malformed_spawn_metadata_falls_back_to_in_process() {
    for value in [json!("nonsense"), json!({"kind": "bogus"}), json!(42)] {
        let spec = with_metadata(one_teammate(), SPAWN_METADATA_KEY, value.clone());
        assert_eq!(
            spawn_options(&spec).kind,
            MemberSpawnKind::InProcess,
            "spawn metadata {value} did not fall back to in-process"
        );
    }
}
