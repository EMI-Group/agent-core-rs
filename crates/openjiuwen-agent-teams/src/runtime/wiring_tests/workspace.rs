//! `metadata["workspace"]` → `IsolationMode`, with `None` as the default.
//!
//! The runtime resolves the isolation once at build time and stamps it on every
//! [`WorkspaceRequest`](crate::workspace::WorkspaceRequest) it hands to the injected provider: one
//! request for the leader at build, one per `spawn_member`. These cases pin that wiring with a
//! recording provider, so the assertions are about *what the runtime asked for*, not about how a
//! backend would honour it.
//! The real worktree backend (git binary, repository, cleanup) is covered in
//! `src/workspace/worktree_git_tests.rs`; nothing here allocates a real git worktree.
use super::{blueprint, wire, with_metadata};
use crate::runtime::spec_meta::isolation_mode;
use crate::schema::{MemberStatus, TeamAgentSpec};
use crate::workspace::IsolationMode;
use serde_json::{json, Value};
/// Builds a one-teammate team, spawns the teammate and returns the isolation each request asked for.
async fn recorded_modes(metadata: Option<Value>) -> Vec<IsolationMode> {
    let mut spec = blueprint("alpha", "boss", &["dev-1"]);
    if let Some(value) = metadata {
        spec = with_metadata(spec, "workspace", value);
    }
    let wiring = wire(&spec, |deps| deps).await;
    wiring.agent.spawn_member("dev-1").await.unwrap();
    wiring
        .workspace
        .requests()
        .iter()
        .map(|request| request.isolation)
        .collect()
}
/// Builds a bare blueprint carrying `metadata["workspace"] = value` (or nothing).
fn spec_with(value: Option<Value>) -> TeamAgentSpec {
    let spec = blueprint("alpha", "boss", &["dev-1"]);
    match value {
        Some(value) => with_metadata(spec, "workspace", value),
        None => spec,
    }
}
#[tokio::test]
async fn the_default_isolation_is_none_for_leader_and_teammate() {
    let spec = blueprint("alpha", "boss", &["dev-1"]);
    let wiring = wire(&spec, |deps| deps).await;
    wiring.agent.spawn_member("dev-1").await.unwrap();
    let requests = wiring.workspace.requests();
    assert_eq!(requests.len(), 2, "one leader build, one teammate spawn");
    assert_eq!(requests[0].member_name, "boss");
    assert_eq!(requests[1].member_name, "dev-1");
    assert!(
        requests
            .iter()
            .all(|request| request.isolation == IsolationMode::None),
        "the default is local directories: {requests:?}"
    );
}
#[tokio::test]
async fn an_object_with_worktree_reaches_leader_and_teammate() {
    let modes = recorded_modes(Some(json!({"isolation": "worktree"}))).await;
    assert_eq!(modes, vec![IsolationMode::Worktree; 2]);
}
#[tokio::test]
async fn a_bare_worktree_string_is_honoured_too() {
    let modes = recorded_modes(Some(json!("worktree"))).await;
    assert_eq!(
        modes,
        vec![IsolationMode::Worktree; 2],
        "Python accepts `\"worktree\"` as well as the object form"
    );
}
#[tokio::test]
async fn non_worktree_values_stay_on_local_directories() {
    for value in [
        json!({"isolation": "none"}),
        json!({"isolation": ""}),
        json!({"isolation": "bogus"}),
        json!("bogus"),
        Value::Null,
        json!({"isolation": 7}),
    ] {
        let modes = recorded_modes(Some(value.clone())).await;
        assert_eq!(
            modes,
            vec![IsolationMode::None; 2],
            "value {value} must not select a worktree"
        );
    }
}
#[tokio::test]
async fn a_worktree_team_still_builds_and_spawns() {
    let spec = spec_with(Some(json!({"isolation": "worktree"})));
    let wiring = wire(&spec, |deps| deps).await;
    wiring
        .agent
        .spawn_member("dev-1")
        .await
        .unwrap_or_else(|error| panic!("the recording provider ignores the mode: {error}"));
    assert_eq!(
        wiring
            .agent
            .backend()
            .member("dev-1")
            .expect("the spawned row")
            .status(),
        MemberStatus::Ready
    );
    assert_eq!(wiring.workspace.requests().len(), 2);
}
#[test]
fn isolation_mode_reads_every_documented_shape() {
    assert_eq!(isolation_mode(&spec_with(None)), IsolationMode::None);
    assert_eq!(
        isolation_mode(&spec_with(Some(json!("worktree")))),
        IsolationMode::Worktree
    );
    assert_eq!(
        isolation_mode(&spec_with(Some(json!({"isolation": "Worktree"})))),
        IsolationMode::Worktree,
        "the keyword is case-insensitive"
    );
    assert_eq!(
        isolation_mode(&spec_with(Some(json!({"isolation": " worktree "})))),
        IsolationMode::Worktree,
        "surrounding whitespace is trimmed"
    );
    assert_eq!(
        isolation_mode(&spec_with(Some(json!({"isolation": "none"})))),
        IsolationMode::None
    );
    assert_eq!(
        isolation_mode(&spec_with(Some(json!({"isolation": "bogus"})))),
        IsolationMode::None,
        "an unknown mode is logged and ignored"
    );
    assert_eq!(
        isolation_mode(&spec_with(Some(json!({"isolation": ""})))),
        IsolationMode::None
    );
    assert_eq!(
        isolation_mode(&spec_with(Some(Value::Null))),
        IsolationMode::None
    );
    assert_eq!(
        isolation_mode(&spec_with(Some(json!({"isolation": null})))),
        IsolationMode::None
    );
    assert_eq!(
        isolation_mode(&spec_with(Some(json!({"other": "worktree"})))),
        IsolationMode::None,
        "no `isolation` key means no isolation"
    );
}
