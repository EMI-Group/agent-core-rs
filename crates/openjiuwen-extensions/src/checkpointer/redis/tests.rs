//! Unit-test matrix for the Redis checkpointer.
//!
//! Covers save/load/list/latest/delete/release/`session_exists`, TTL handling on
//! save and on read, and the documented deferrals (TLS rejection, cluster
//! redirection).
//! The tests are intentionally kept in one file because they share the
//! `FakeRedis` fixtures and the configuration helpers defined below.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)] // `assert!(x.is_empty())` states the expectation directly

use super::*;
use crate::checkpointer::CheckpointNamespace;
use crate::redis::testsupport::{FakeOptions, FakeRedis};
use openjiuwen_common::error::ErrorKind;
use serde_json::json;
use std::time::Duration;

// port of tests/unit_tests/extensions/checkpointer/test_redis_checkpointer.py and
// test_redis_checkpointer_provider.py

fn valid_config() -> RedisCheckpointerConfig {
    RedisCheckpointerConfig::new(RedisConnectionConfig::new("redis://127.0.0.1:6379"))
}

fn config_for(fake: &FakeRedis) -> RedisCheckpointerConfig {
    RedisCheckpointerConfig::new(RedisConnectionConfig::new(fake.url()))
}

fn ttl_config_for(
    fake: &FakeRedis,
    minutes: f64,
    refresh_on_read: bool,
) -> RedisCheckpointerConfig {
    config_for(fake).with_ttl(Some(RedisTTLConfig::new(Some(minutes), refresh_on_read)))
}

async fn connect(fake: &FakeRedis) -> Arc<dyn Checkpointer> {
    RedisCheckpointer::connect(config_for(fake)).await.unwrap()
}

#[test]
fn accepts_every_python_scheme() {
    for url in [
        "redis://localhost:6379",
        "rediss://localhost:6379",
        "redis+cluster://localhost:7000",
        "rediss+cluster://localhost:7000",
    ] {
        let config = RedisConnectionConfig::new(url);
        assert!(config.validate().is_ok(), "{url} must be accepted");
    }
}

#[test]
fn rejects_missing_or_foreign_urls() {
    let missing = RedisConnectionConfig::default();
    let error = missing.validate().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("url"), "unexpected: {error}");

    let foreign = RedisConnectionConfig::new("http://localhost:6379");
    let error = foreign.validate().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.to_string().contains("invalid Redis URL format"),
        "unexpected: {error}"
    );
}

#[test]
fn cluster_mode_is_inferred_from_the_scheme_when_unset() {
    assert!(!RedisConnectionConfig::new("redis://localhost:6379").is_cluster_mode());
    assert!(!RedisConnectionConfig::new("rediss://localhost:6379").is_cluster_mode());
    assert!(RedisConnectionConfig::new("redis+cluster://localhost:7000").is_cluster_mode());
    assert!(RedisConnectionConfig::new("rediss+cluster://localhost:7000").is_cluster_mode());
}

#[test]
fn explicit_cluster_mode_overrides_the_scheme() {
    let mut config = RedisConnectionConfig::new("redis://localhost:6379");
    config.cluster_mode = Some(true);
    assert!(config.is_cluster_mode());

    let mut config = RedisConnectionConfig::new("redis+cluster://localhost:7000");
    config.cluster_mode = Some(false);
    assert!(!config.is_cluster_mode());
}

#[test]
fn connection_url_strips_the_cluster_marker() {
    assert_eq!(
        RedisConnectionConfig::new("redis+cluster://localhost:7000").connection_url(),
        Some("redis://localhost:7000".to_string())
    );
    assert_eq!(
        RedisConnectionConfig::new("rediss+cluster://localhost:7000").connection_url(),
        Some("rediss://localhost:7000".to_string())
    );
    assert_eq!(
        RedisConnectionConfig::new("redis://localhost:6379").connection_url(),
        Some("redis://localhost:6379".to_string())
    );
    assert_eq!(RedisConnectionConfig::default().connection_url(), None);
}

#[test]
fn connection_args_are_carried_through() {
    let config: RedisConnectionConfig = serde_json::from_value(json!({
        "url": "redis://localhost:6379",
        "connection_args": {"db": 1, "timeout": 5}
    }))
    .unwrap();
    assert_eq!(config.connection_args["db"], json!(1));
    assert_eq!(config.connection_args["timeout"], json!(5));
    assert!(!config.is_cluster_mode());
}

#[test]
fn checkpointer_config_round_trips_with_python_field_names() {
    let value = json!({
        "connection": {"url": "redis://localhost:6379", "cluster_mode": true},
        "ttl": {"default_ttl": 5.0, "refresh_on_read": true}
    });
    let config: RedisCheckpointerConfig = serde_json::from_value(value).unwrap();
    assert_eq!(
        config.connection.url.as_deref(),
        Some("redis://localhost:6379")
    );
    assert!(config.connection.is_cluster_mode());
    let ttl = config.ttl.as_ref().unwrap();
    assert_eq!(ttl.default_ttl_minutes, Some(5.0));
    assert!(ttl.refresh_on_read);

    let encoded = serde_json::to_value(&config).unwrap();
    assert_eq!(
        encoded["connection"]["url"],
        json!("redis://localhost:6379")
    );
    assert_eq!(encoded["ttl"]["default_ttl"], json!(5.0));
    assert_eq!(encoded["ttl"]["refresh_on_read"], json!(true));
}

#[test]
fn new_validates_the_connection_url() {
    let error = RedisCheckpointer::new(RedisCheckpointerConfig::new(RedisConnectionConfig::new(
        "noscheme",
    )))
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);

    let checkpointer = RedisCheckpointer::new(valid_config()).unwrap();
    assert_eq!(
        checkpointer.config().connection.url.as_deref(),
        Some("redis://127.0.0.1:6379")
    );
}

#[test]
fn release_patterns_escape_glob_metacharacters() {
    assert_eq!(escape_glob("plain-1"), "plain-1");
    assert_eq!(escape_glob("a*b?c[d]e\\f"), "a\\*b\\?c\\[d\\]e\\\\f");
}

#[test]
fn ttl_seconds_rounds_up_and_never_drops_below_one_second() {
    assert_eq!(ttl_seconds(5.0), 300);
    assert_eq!(ttl_seconds(0.001), 1);
    assert_eq!(ttl_seconds(0.05), 3);
    assert_eq!(ttl_seconds(0.0), 1);
    assert_eq!(ttl_seconds(-2.0), 1);
    assert_eq!(ttl_seconds(f64::NAN), 1);
}

#[test]
fn debug_output_never_leaks_credentials() {
    let mut config = RedisCheckpointerConfig::new(RedisConnectionConfig::new(
        "redis://user:secret@127.0.0.1:6379",
    ));
    config
        .connection
        .connection_args
        .insert("password".to_owned(), json!("secret"));
    let checkpointer = RedisCheckpointer::new(config).unwrap();
    let rendered = format!("{checkpointer:?}");
    assert!(
        !rendered.contains("secret"),
        "unexpected debug output: {rendered}"
    );
}

#[tokio::test]
async fn save_and_load_round_trip_every_field() {
    let fake = FakeRedis::start().await;
    let checkpointer = connect(&fake).await;
    let mut metadata = BTreeMap::new();
    metadata.insert("origin".to_owned(), json!("test"));
    let checkpoint = Checkpoint::new("session-1", CheckpointNamespace::AgentTeam, "team-1")
        .with_global_state(json!({"phase": "running"}))
        .with_component_state(json!({"component-1": {"done": false}}))
        .with_updates(json!({"step": 3}))
        .with_metadata(metadata);

    let checkpoint_id = checkpointer.save(checkpoint).await.unwrap();
    assert!(!checkpoint_id.is_empty());

    let loaded = checkpointer
        .load("session-1", &checkpoint_id)
        .await
        .unwrap()
        .expect("the checkpoint was just saved");
    assert_eq!(loaded.session_id, "session-1");
    assert_eq!(loaded.checkpoint_id, checkpoint_id);
    assert_eq!(loaded.namespace, CheckpointNamespace::AgentTeam);
    assert_eq!(loaded.entity_id, "team-1");
    assert!(!loaded.created_at.is_empty());
    assert_eq!(loaded.global_state, json!({"phase": "running"}));
    assert_eq!(
        loaded.component_state,
        json!({"component-1": {"done": false}})
    );
    assert_eq!(loaded.updates, json!({"step": 3}));
    assert_eq!(loaded.metadata["origin"], json!("test"));
}

#[tokio::test]
async fn save_honours_an_explicit_checkpoint_id() {
    let fake = FakeRedis::start().await;
    let state = fake.state();
    let checkpointer = connect(&fake).await;
    let checkpoint = Checkpoint::new("session-1", CheckpointNamespace::Agent, "agent-1")
        .with_checkpoint_id("cp-explicit");

    assert_eq!(checkpointer.save(checkpoint).await.unwrap(), "cp-explicit");
    assert!(checkpointer
        .load("session-1", "cp-explicit")
        .await
        .unwrap()
        .is_some());
    assert_eq!(state.zset(&index_key("session-1")).len(), 1);
}

#[tokio::test]
async fn invalid_ids_are_validation_errors() {
    let fake = FakeRedis::start().await;
    let checkpointer = connect(&fake).await;

    let error = checkpointer
        .save(Checkpoint::new(
            "a/b",
            CheckpointNamespace::Agent,
            "agent-1",
        ))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);

    let error = checkpointer
        .save(
            Checkpoint::new("session-1", CheckpointNamespace::Agent, "agent-1")
                .with_checkpoint_id(".."),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);

    for (session_id, checkpoint_id) in [("a/b", "cp-1"), ("session-1", "..")] {
        assert_eq!(
            checkpointer
                .load(session_id, checkpoint_id)
                .await
                .unwrap_err()
                .kind(),
            ErrorKind::Validation
        );
        assert_eq!(
            checkpointer
                .delete(session_id, checkpoint_id)
                .await
                .unwrap_err()
                .kind(),
            ErrorKind::Validation
        );
    }
    assert_eq!(
        checkpointer.list("..").await.unwrap_err().kind(),
        ErrorKind::Validation
    );
    assert_eq!(
        checkpointer.release("a/b").await.unwrap_err().kind(),
        ErrorKind::Validation
    );
    assert_eq!(
        checkpointer.session_exists("..").await.unwrap_err().kind(),
        ErrorKind::Validation
    );
    assert!(fake.state().keys().is_empty());
}

#[tokio::test]
async fn missing_sessions_and_checkpoints_are_empty_results() {
    let fake = FakeRedis::start().await;
    let checkpointer = connect(&fake).await;

    assert!(checkpointer
        .load("session-1", "cp-1")
        .await
        .unwrap()
        .is_none());
    assert!(checkpointer.list("session-1").await.unwrap().is_empty());
    assert!(!checkpointer.session_exists("session-1").await.unwrap());
    assert!(checkpointer.latest("session-1").await.unwrap().is_none());

    let checkpoint_id = checkpointer
        .save(Checkpoint::new(
            "session-1",
            CheckpointNamespace::Agent,
            "agent-1",
        ))
        .await
        .unwrap();
    assert!(checkpointer.session_exists("session-1").await.unwrap());

    checkpointer.release("session-1").await.unwrap();
    assert!(!checkpointer.session_exists("session-1").await.unwrap());
    assert!(checkpointer
        .load("session-1", &checkpoint_id)
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn list_returns_newest_first_and_latest_picks_the_newest() {
    let fake = FakeRedis::start().await;
    let checkpointer = connect(&fake).await;
    for (checkpoint_id, created_at) in [
        ("cp-b", "2024-01-02T00:00:00Z"),
        ("cp-a", "2024-01-01T00:00:00Z"),
        ("cp-c", "2024-01-03T00:00:00Z"),
        ("cp-d", "2024-01-04T00:00:00Z"),
        ("cp-e", "2024-01-04T00:00:00Z"),
    ] {
        let mut checkpoint = Checkpoint::new("session-1", CheckpointNamespace::Workflow, "flow-1")
            .with_checkpoint_id(checkpoint_id);
        checkpoint.created_at = created_at.to_owned();
        checkpointer.save(checkpoint).await.unwrap();
    }

    let ids: Vec<String> = checkpointer
        .list("session-1")
        .await
        .unwrap()
        .into_iter()
        .map(|meta| meta.checkpoint_id)
        .collect();
    assert_eq!(
        ids,
        vec![
            "cp-e".to_owned(),
            "cp-d".to_owned(),
            "cp-c".to_owned(),
            "cp-b".to_owned(),
            "cp-a".to_owned(),
        ]
    );
    assert_eq!(
        checkpointer
            .latest("session-1")
            .await
            .unwrap()
            .unwrap()
            .checkpoint_id,
        "cp-e"
    );
}

#[tokio::test]
async fn delete_removes_one_checkpoint_and_keeps_the_session() {
    let fake = FakeRedis::start().await;
    let state = fake.state();
    let checkpointer = connect(&fake).await;
    let first = checkpointer
        .save(Checkpoint::new(
            "session-1",
            CheckpointNamespace::Agent,
            "agent-1",
        ))
        .await
        .unwrap();
    let second = checkpointer
        .save(Checkpoint::new(
            "session-1",
            CheckpointNamespace::Agent,
            "agent-1",
        ))
        .await
        .unwrap();

    checkpointer.delete("session-1", &first).await.unwrap();
    assert!(checkpointer
        .load("session-1", &first)
        .await
        .unwrap()
        .is_none());
    assert!(checkpointer
        .load("session-1", &second)
        .await
        .unwrap()
        .is_some());
    assert!(checkpointer.session_exists("session-1").await.unwrap());
    assert_eq!(state.zset(&index_key("session-1")).len(), 1);

    // Deleting a missing checkpoint or from an unknown session is a no-op.
    checkpointer.delete("session-1", &first).await.unwrap();
    checkpointer
        .delete("session-2", "cp-missing")
        .await
        .unwrap();
    assert!(checkpointer
        .load("session-1", &second)
        .await
        .unwrap()
        .is_some());
}

#[tokio::test]
async fn release_removes_payloads_and_the_index() {
    let fake = FakeRedis::start().await;
    let state = fake.state();
    let checkpointer = connect(&fake).await;
    let first = checkpointer
        .save(Checkpoint::new(
            "session-1",
            CheckpointNamespace::Agent,
            "agent-1",
        ))
        .await
        .unwrap();
    let second = checkpointer
        .save(Checkpoint::new(
            "session-1",
            CheckpointNamespace::Agent,
            "agent-1",
        ))
        .await
        .unwrap();
    let neighbour = checkpointer
        .save(Checkpoint::new(
            "session-10",
            CheckpointNamespace::Agent,
            "agent-1",
        ))
        .await
        .unwrap();

    checkpointer.release("session-1").await.unwrap();

    let remaining = state.keys();
    assert!(!remaining
        .iter()
        .any(|key| key.starts_with(&format!("{PAYLOAD_PREFIX}session-1:"))));
    assert!(!remaining.iter().any(|key| key == &index_key("session-1")));
    // The release pattern ends in `:` so a longer id sharing the prefix survives.
    assert!(remaining
        .iter()
        .any(|key| key == &payload_key("session-10", &neighbour)));
    assert!(checkpointer
        .load("session-10", &neighbour)
        .await
        .unwrap()
        .is_some());
    assert!(checkpointer
        .load("session-1", &first)
        .await
        .unwrap()
        .is_none());
    assert!(checkpointer
        .load("session-1", &second)
        .await
        .unwrap()
        .is_none());

    // Releasing an unknown session is a no-op.
    checkpointer.release("session-2").await.unwrap();
}

#[tokio::test]
async fn list_drops_index_entries_without_a_payload() {
    let fake = FakeRedis::start().await;
    let state = fake.state();
    let checkpointer = connect(&fake).await;
    let checkpoint_id = checkpointer
        .save(Checkpoint::new(
            "session-1",
            CheckpointNamespace::Agent,
            "agent-1",
        ))
        .await
        .unwrap();

    let client = Client::from_url(&fake.url()).await.unwrap();
    client
        .command_str(&["DEL", &payload_key("session-1", &checkpoint_id)])
        .await
        .unwrap();

    assert!(checkpointer.list("session-1").await.unwrap().is_empty());
    assert!(
        state.zset(&index_key("session-1")).is_empty(),
        "the stale index entry must be dropped"
    );
}

#[tokio::test]
async fn undecodable_payloads_are_session_errors() {
    let fake = FakeRedis::start().await;
    let checkpointer = connect(&fake).await;
    let client = Client::from_url(&fake.url()).await.unwrap();
    client
        .command_str(&["SET", &payload_key("session-1", "cp-broken"), "not json"])
        .await
        .unwrap();

    let error = checkpointer
        .load("session-1", "cp-broken")
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Session);
    assert!(
        !error.to_string().contains("not json"),
        "the payload must not leak: {error}"
    );
    assert!(
        error.to_string().contains("cp-broken"),
        "unexpected: {error}"
    );
}

#[tokio::test]
async fn without_a_ttl_config_keys_never_expire() {
    let fake = FakeRedis::start().await;
    let state = fake.state();
    let checkpointer = connect(&fake).await;
    let checkpoint_id = checkpointer
        .save(Checkpoint::new(
            "session-1",
            CheckpointNamespace::Agent,
            "agent-1",
        ))
        .await
        .unwrap();

    assert_eq!(
        state.ttl_ms(&payload_key("session-1", &checkpoint_id)),
        Some(-1)
    );
    assert_eq!(state.ttl_ms(&index_key("session-1")), Some(-1));
}

#[tokio::test]
async fn a_configured_ttl_expires_the_payload_and_the_index() {
    let fake = FakeRedis::start().await;
    let state = fake.state();
    let checkpointer = RedisCheckpointer::connect(ttl_config_for(&fake, 5.0, false))
        .await
        .unwrap();
    let checkpoint_id = checkpointer
        .save(Checkpoint::new(
            "session-1",
            CheckpointNamespace::Agent,
            "agent-1",
        ))
        .await
        .unwrap();

    let payload_ttl = state
        .ttl_ms(&payload_key("session-1", &checkpoint_id))
        .expect("the payload key exists");
    assert!(
        (250_000..=300_000).contains(&payload_ttl),
        "unexpected payload ttl: {payload_ttl}"
    );
    let index_ttl = state
        .ttl_ms(&index_key("session-1"))
        .expect("the index key exists");
    assert!(
        (250_000..=300_000).contains(&index_ttl),
        "unexpected index ttl: {index_ttl}"
    );
}

#[tokio::test]
async fn refresh_on_read_rearms_the_payload_and_index_ttls() {
    let fake = FakeRedis::start().await;
    let state = fake.state();
    // 0.05 minutes = 3 seconds, short enough to observe the decay.
    let checkpointer = RedisCheckpointer::connect(ttl_config_for(&fake, 0.05, true))
        .await
        .unwrap();
    let checkpoint_id = checkpointer
        .save(Checkpoint::new(
            "session-1",
            CheckpointNamespace::Agent,
            "agent-1",
        ))
        .await
        .unwrap();
    let key = payload_key("session-1", &checkpoint_id);

    tokio::time::sleep(Duration::from_millis(200)).await;
    let decayed = state.ttl_ms(&key).expect("the payload key exists");
    assert!(decayed > 0, "the payload key expired too early: {decayed}");

    assert!(checkpointer
        .load("session-1", &checkpoint_id)
        .await
        .unwrap()
        .is_some());
    let refreshed = state.ttl_ms(&key).expect("the payload key exists");
    assert!(
        refreshed > decayed + 100,
        "the read did not refresh the ttl: {decayed} -> {refreshed}"
    );
    assert!(
        state
            .ttl_ms(&index_key("session-1"))
            .is_some_and(|ttl| ttl > decayed + 100),
        "the index ttl was not refreshed"
    );
}

#[tokio::test]
async fn a_read_without_refresh_does_not_extend_the_ttl() {
    let fake = FakeRedis::start().await;
    let state = fake.state();
    let checkpointer = RedisCheckpointer::connect(ttl_config_for(&fake, 0.05, false))
        .await
        .unwrap();
    let checkpoint_id = checkpointer
        .save(Checkpoint::new(
            "session-1",
            CheckpointNamespace::Agent,
            "agent-1",
        ))
        .await
        .unwrap();
    let key = payload_key("session-1", &checkpoint_id);

    tokio::time::sleep(Duration::from_millis(100)).await;
    let decayed = state.ttl_ms(&key).expect("the payload key exists");
    assert!(checkpointer
        .load("session-1", &checkpoint_id)
        .await
        .unwrap()
        .is_some());
    let after_read = state.ttl_ms(&key).expect("the payload key exists");
    assert!(
        after_read <= decayed,
        "the read must not extend the ttl: {decayed} -> {after_read}"
    );
}

#[tokio::test]
async fn connect_and_create_reach_a_live_server() {
    let fake = FakeRedis::start().await;
    let checkpointer = RedisCheckpointer::connect(config_for(&fake)).await.unwrap();
    let checkpoint_id = checkpointer
        .save(Checkpoint::new(
            "session-1",
            CheckpointNamespace::Agent,
            "agent-1",
        ))
        .await
        .unwrap();
    assert!(checkpointer
        .load("session-1", &checkpoint_id)
        .await
        .unwrap()
        .is_some());

    let created = RedisCheckpointer::create(&config_for(&fake)).await.unwrap();
    assert!(created
        .load("session-1", &checkpoint_id)
        .await
        .unwrap()
        .is_some());
    assert!(fake.state().connection_count() >= 1);
}

#[tokio::test]
async fn tls_urls_are_unsupported() {
    let error = RedisCheckpointer::connect(RedisCheckpointerConfig::new(
        RedisConnectionConfig::new("rediss://127.0.0.1:6379"),
    ))
    .await
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(
        error.to_string().to_lowercase().contains("tls"),
        "unexpected: {error}"
    );

    // The cluster TLS scheme is normalised to `rediss://` and fails the same way.
    let error = RedisCheckpointer::create(&RedisCheckpointerConfig::new(
        RedisConnectionConfig::new("rediss+cluster://127.0.0.1:7000"),
    ))
    .await
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
}

#[tokio::test]
async fn cluster_mode_against_a_redirecting_server_is_unsupported() {
    let fake = FakeRedis::start_with(FakeOptions::moved_everything()).await;
    let config = RedisCheckpointerConfig::new(RedisConnectionConfig::new(format!(
        "redis+cluster://{}",
        fake.addr()
    )));

    let error = RedisCheckpointer::connect(config).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    let message = error.to_string();
    assert!(
        message.contains("cluster") && message.contains("MOVED"),
        "unexpected: {message}"
    );
}

#[tokio::test]
async fn an_unreachable_server_is_a_transport_error() {
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);

    let error = RedisCheckpointer::connect(RedisCheckpointerConfig::new(
        RedisConnectionConfig::new(format!("redis://{addr}")),
    ))
    .await
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Transport);
}

#[tokio::test]
async fn credentials_from_connection_args_are_applied() {
    let fake = FakeRedis::start_with(FakeOptions::auth_required(Some("user"), "pass")).await;
    let mut config = config_for(&fake);
    config
        .connection
        .connection_args
        .insert("username".to_owned(), json!("user"));
    config
        .connection
        .connection_args
        .insert("password".to_owned(), json!("pass"));

    let checkpointer = RedisCheckpointer::connect(config).await.unwrap();
    let checkpoint_id = checkpointer
        .save(Checkpoint::new(
            "session-1",
            CheckpointNamespace::Agent,
            "agent-1",
        ))
        .await
        .unwrap();
    assert!(checkpointer
        .load("session-1", &checkpoint_id)
        .await
        .unwrap()
        .is_some());
    assert!(fake.state().auth_attempts() >= 1);
}
