//! KV-cache affinity: register → env → release across a member lifecycle, forks included.
//!
//! The seam under test is `TeamAgentSpec::metadata["kv_cache"]` (default: enabled) plus the
//! affinity calls the build path, `spawn_member`, `shutdown_member` and `stop` make:
//!
//! | lifecycle fact | registry effect |
//! | --- | --- |
//! | `build_with` | the leader is registered as a root member |
//! | `spawn_member` | the teammate is registered, as a fork child when a directive applied |
//! | `shutdown_member` | `release(member)` drops that member's rows |
//! | `stop` | `release_team(session)` drops every row of the team |
//!
//! A root member's identity is parented at the *team session*, so both documented env keys are
//! always present; a forked member's parent is its parent member's cache id instead.

use std::collections::BTreeMap;
use std::sync::Arc;

use openjiuwen_agent::kv_cache::KVCacheAffinityConfig;
use openjiuwen_common::error::ErrorKind;
use serde_json::json;

use crate::kv_cache::{MemberCacheKey, TeamKvCacheAffinity};
use crate::schema::{MemberLifetime, MemberStatus};

use super::{blueprint, wire, with_metadata};

/// Env key carrying a member's cache id.
const SESSION_ENV: &str = "kv_cache_affinity_session_id";
/// Env key carrying a member's lineage (the team session for a root member).
const PARENT_ENV: &str = "kv_cache_affinity_parent_session_id";

/// Returns the cache id carried by `env`, failing the test when the key is absent.
fn session_of(env: &BTreeMap<String, String>) -> &str {
    env.get(SESSION_ENV)
        .map(String::as_str)
        .unwrap_or_else(|| panic!("no '{SESSION_ENV}' in {env:?}"))
}

/// Returns the lineage cache id carried by `env`, failing the test when the key is absent.
fn parent_of(env: &BTreeMap<String, String>) -> &str {
    env.get(PARENT_ENV)
        .map(String::as_str)
        .unwrap_or_else(|| panic!("no '{PARENT_ENV}' in {env:?}"))
}

#[tokio::test]
async fn fresh_build_registers_the_leader_as_a_root_member() {
    let wiring = wire(&blueprint("alpha", "boss", &["dev-1"]), |deps| deps).await;
    let agent = &wiring.agent;

    assert!(agent.kv_cache().is_enabled(), "affinity is on by default");
    let env = agent
        .kv_env("boss")
        .expect("the leader has no affinity env after the build");

    // Exactly the two documented keys: the member cache id and its lineage at the team session.
    assert_eq!(env.len(), 2, "{env:?}");
    assert_eq!(
        session_of(&env),
        format!("team:{}:team:alpha:member:boss", agent.session_id())
    );
    assert_eq!(
        parent_of(&env),
        agent.session_id(),
        "a root member is parented at the team session"
    );

    assert_eq!(agent.kv_cache().len(), 1);
    assert_eq!(agent.kv_cache().members(), vec!["boss"]);
    assert_eq!(
        agent.kv_cache().identity("boss").map(|id| id.session_id),
        Some(session_of(&env).to_owned())
    );
    assert!(
        agent.kv_env("dev-1").is_none(),
        "a teammate registers only once it is spawned"
    );
}

#[tokio::test]
async fn spawn_registers_a_teammate_identity() {
    let wiring = wire(&blueprint("alpha", "boss", &["dev-1"]), |deps| deps).await;
    let agent = &wiring.agent;

    agent.spawn_member("dev-1").await.unwrap();

    let env = agent
        .kv_env("dev-1")
        .expect("the spawned teammate has no affinity env");
    assert_eq!(env.len(), 2, "{env:?}");
    assert_eq!(
        session_of(&env),
        format!("team:{}:team:alpha:member:dev-1", agent.session_id())
    );
    assert_eq!(parent_of(&env), agent.session_id());

    assert_eq!(agent.kv_cache().len(), 2);
    assert!(agent.kv_cache().identity("dev-1").is_some());
    assert!(
        agent
            .kv_cache()
            .entry("dev-1")
            .unwrap()
            .parent_member
            .is_none(),
        "a plain teammate is a root member, not a fork"
    );

    let leader = agent
        .kv_env("boss")
        .expect("the leader lost its affinity env");
    assert_ne!(
        session_of(&env),
        session_of(&leader),
        "each member owns its own cache id"
    );
}

#[tokio::test]
async fn disabled_affinity_registers_nothing() {
    let spec = with_metadata(
        blueprint("alpha", "boss", &["dev-1"]),
        "kv_cache",
        json!({ "enabled": false }),
    );
    let wiring = wire(&spec, |deps| deps).await;
    let agent = &wiring.agent;

    assert!(!agent.kv_cache().is_enabled());
    agent.spawn_member("dev-1").await.unwrap();

    for name in ["boss", "dev-1"] {
        assert!(agent.kv_env(name).is_none(), "{name} has an affinity env");
    }
    assert_eq!(agent.kv_cache().len(), 0);
    assert!(agent.kv_cache().is_empty());
    assert!(agent.kv_cache().members().is_empty());

    let error = agent
        .kv_cache()
        .register_child(
            "boss",
            MemberCacheKey::new(agent.team_name(), agent.session_id(), "child"),
            "gpt-test",
        )
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(
        error.to_string().contains("kv-cache affinity is disabled"),
        "{error}"
    );
}

#[tokio::test]
async fn injected_registry_is_reused_by_the_build() {
    let shared = Arc::new(TeamKvCacheAffinity::new(KVCacheAffinityConfig::enabled()));
    let wiring = wire(&blueprint("alpha", "boss", &[]), |deps| {
        deps.with_kv_cache(Arc::clone(&shared))
    })
    .await;

    assert!(
        Arc::ptr_eq(&wiring.agent.kv_cache(), &shared),
        "the build must reuse the injected registry"
    );
    assert_eq!(
        shared.len(),
        1,
        "the leader registered in the shared registry"
    );
    assert!(shared.identity("boss").is_some());
    assert_eq!(
        wiring.agent.kv_env("boss").map(|env| env.len()),
        Some(2),
        "the agent reads the same registry the host injected"
    );
}

#[tokio::test]
async fn forked_member_inherits_the_parent_identity() {
    let spec = with_metadata(
        blueprint("alpha", "boss", &["dev-1", "dev-2"]),
        "fork",
        json!({ "parent": "dev-1" }),
    );
    let wiring = wire(&spec, |deps| deps).await;
    let agent = &wiring.agent;

    assert_eq!(
        agent
            .fork_directive()
            .map(|directive| directive.parent.as_str()),
        Some("dev-1")
    );

    agent.spawn_member("dev-1").await.unwrap();
    agent.spawn_member("dev-2").await.unwrap();

    let forked = agent
        .spawner()
        .forked_member("dev-2")
        .expect("dev-2 was not forked");
    assert_eq!(forked.parent_member, "dev-1");
    assert!(
        agent.spawner().forked_member("dev-1").is_none(),
        "a directive naming the member itself never forks it"
    );

    let parent = agent.kv_env("dev-1").expect("dev-1 has no affinity env");
    let child = agent.kv_env("dev-2").expect("dev-2 has no affinity env");
    assert_eq!(
        child.get(PARENT_ENV),
        parent.get(SESSION_ENV),
        "the child's lineage is the parent's cache id"
    );
    assert_ne!(
        child.get(SESSION_ENV),
        parent.get(SESSION_ENV),
        "the child owns its own cache id"
    );
    assert!(
        session_of(&child).ends_with(":fork:dev-2"),
        "{}",
        session_of(&child)
    );
    assert_eq!(agent.kv_cache().len(), 3);
}

#[tokio::test]
async fn shutdown_releases_one_member_and_stop_releases_the_team() {
    let wiring = wire(&blueprint("alpha", "boss", &["dev-1", "dev-2"]), |deps| {
        deps
    })
    .await;
    let agent = &wiring.agent;

    agent.spawn_member("dev-1").await.unwrap();
    assert_eq!(agent.kv_cache().len(), 2);

    let status = agent
        .shutdown_member("dev-1", MemberLifetime::Shutdown, true)
        .await
        .unwrap();
    assert_eq!(status, MemberStatus::ShutDown);
    assert!(
        agent.kv_env("dev-1").is_none(),
        "a shut-down member keeps no affinity env"
    );
    assert_eq!(agent.kv_cache().len(), 1);
    assert_eq!(agent.kv_cache().members(), vec!["boss"]);

    // The next spawn registers again, from scratch.
    agent.spawn_member("dev-2").await.unwrap();
    assert_eq!(agent.kv_cache().len(), 2);
    assert!(agent.kv_env("dev-2").is_some());

    agent.start().await.unwrap();
    agent.stop().await.unwrap();

    assert_eq!(agent.kv_cache().len(), 0, "stop releases the whole team");
    assert!(agent.kv_cache().is_empty());
    assert!(agent.kv_env("boss").is_none());
    assert!(agent.kv_env("dev-2").is_none());
}
