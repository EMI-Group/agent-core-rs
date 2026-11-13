#![allow(clippy::unwrap_used, clippy::expect_used)]
use super::*;
use crate::redis::testsupport::{FakeOptions, FakeRedis};
use openjiuwen_common::error::ErrorKind;
use serde_json::json;
use std::time::Duration;
fn args(value: serde_json::Value) -> BTreeMap<String, serde_json::Value> {
    serde_json::from_value(value).expect("test JSON is an object")
}
#[test]
fn default_options_target_local_redis() {
    let options = ConnectionOptions::default();
    assert_eq!(options.host, "127.0.0.1");
    assert_eq!(options.port, 6379);
    assert_eq!(options.db, None);
    assert_eq!(options.timeout, Duration::from_secs(10));
    assert!(!options.cluster_mode);
}
#[test]
fn parses_minimal_url() {
    let options = ConnectionOptions::from_url("redis://localhost").unwrap();
    assert_eq!(options.host, "localhost");
    assert_eq!(options.port, 6379);
    assert_eq!(options.db, None);
    assert!(!options.cluster_mode);
    assert!(options.username.is_none());
    assert!(options.password.is_none());
}
#[test]
fn parses_credentials_and_database() {
    let options =
        ConnectionOptions::from_url("redis://alice:p%40ss%3Aword@example.com:6380/3").unwrap();
    assert_eq!(options.host, "example.com");
    assert_eq!(options.port, 6380);
    assert_eq!(options.username.as_deref(), Some("alice"));
    assert_eq!(options.password.as_deref(), Some("p@ss:word"));
    assert_eq!(options.db, Some(3));
}
#[test]
fn parses_password_only_userinfo() {
    let options = ConnectionOptions::from_url("redis://:secret@example.com/0").unwrap();
    assert_eq!(options.username, None);
    assert_eq!(options.password.as_deref(), Some("secret"));
    assert_eq!(options.db, Some(0));
}
#[test]
fn parses_ipv6_host_without_brackets() {
    let options = ConnectionOptions::from_url("redis://[::1]:6380/2").unwrap();
    assert_eq!(options.host, "::1");
    assert_eq!(options.port, 6380);
    assert_eq!(options.db, Some(2));
}
#[test]
fn cluster_scheme_sets_cluster_mode() {
    let options = ConnectionOptions::from_url("redis+cluster://example.com:7000").unwrap();
    assert!(options.cluster_mode);
    assert_eq!(options.port, 7000);
}
#[test]
fn tls_scheme_is_unsupported() {
    for url in ["rediss://example.com", "rediss+cluster://example.com"] {
        let error = ConnectionOptions::from_url(url).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported, "url {url}");
        assert!(error.to_string().contains("TLS"), "url {url}");
    }
}
#[test]
fn other_schemes_and_bad_databases_are_validation_errors() {
    let error = ConnectionOptions::from_url("http://example.com").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("redis://"));
    for url in ["redis://example.com/abc", "redis://example.com/-1"] {
        assert_eq!(
            ConnectionOptions::from_url(url).unwrap_err().kind(),
            ErrorKind::Validation,
            "url {url}"
        );
    }
}
#[test]
fn connection_args_map_onto_options() {
    let options = ConnectionOptions::default()
        .with_connection_args(&args(json!({
            "username": "alice",
            "password": "secret",
            "db": 4,
            "timeout": 2,
            "unrelated": true
        })))
        .unwrap();
    assert_eq!(options.username.as_deref(), Some("alice"));
    assert_eq!(options.password.as_deref(), Some("secret"));
    assert_eq!(options.db, Some(4));
    assert_eq!(options.timeout, Duration::from_secs(2));
}
#[test]
fn connection_args_accept_fractional_and_string_timeouts() {
    let fractional = ConnectionOptions::default()
        .with_connection_args(&args(json!({"timeout": 0.5})))
        .unwrap();
    assert_eq!(fractional.timeout, Duration::from_millis(500));
    let textual = ConnectionOptions::default()
        .with_connection_args(&args(json!({"timeout": "3", "db": "2"})))
        .unwrap();
    assert_eq!(textual.timeout, Duration::from_secs(3));
    assert_eq!(textual.db, Some(2));
}
#[test]
fn connection_args_null_clears_and_invalid_values_fail() {
    let cleared = ConnectionOptions::from_url("redis://user:pw@example.com/1")
        .unwrap()
        .with_connection_args(&args(
            json!({"username": null, "password": null, "db": null}),
        ))
        .unwrap();
    assert!(cleared.username.is_none());
    assert!(cleared.password.is_none());
    assert!(cleared.db.is_none());
    for invalid in [
        json!({"timeout": "soon"}),
        json!({"timeout": 0}),
        json!({"timeout": -1}),
        json!({"db": -3}),
        json!({"username": 5}),
        json!({"password": false}),
    ] {
        let error = ConnectionOptions::default()
            .with_connection_args(&args(invalid.clone()))
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "args {invalid}");
    }
}
#[test]
fn with_cluster_mode_sets_the_flag() {
    let options = ConnectionOptions::default().with_cluster_mode(true);
    assert!(options.cluster_mode);
}
#[test]
fn client_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Client>();
}
#[tokio::test]
async fn round_trips_strings_scans_and_existence() {
    let fake = FakeRedis::start().await;
    let client = Client::from_url(&fake.url()).await.unwrap();
    client.ping().await.unwrap();
    assert_eq!(
        client.command_str(&["SET", "user:1", "ada"]).await.unwrap(),
        Reply::Simple("OK".to_owned())
    );
    assert_eq!(
        client.command_str(&["GET", "user:1"]).await.unwrap(),
        Reply::Bulk(b"ada".to_vec())
    );
    assert_eq!(fake.state().get("user:1"), Some(b"ada".to_vec()));
    assert_eq!(fake.state().keys(), vec!["user:1".to_owned()]);
    assert!(fake.state().connection_count() >= 1);
    assert_eq!(
        client
            .command_str(&["EXISTS", "user:1", "missing"])
            .await
            .unwrap(),
        Reply::Integer(1)
    );
    assert_eq!(
        client
            .command_str(&["MGET", "user:1", "missing"])
            .await
            .unwrap(),
        Reply::Array(vec![Reply::Bulk(b"ada".to_vec()), Reply::Null])
    );
    let scan = client
        .command_str(&["SCAN", "0", "MATCH", "user:*", "COUNT", "10"])
        .await
        .unwrap();
    let scan = scan.as_array().unwrap();
    assert_eq!(scan[0], Reply::Bulk(b"0".to_vec()));
    assert_eq!(
        scan[1].as_array().unwrap(),
        [Reply::Bulk(b"user:1".to_vec())]
    );
    assert_eq!(
        client.command_str(&["DEL", "user:1"]).await.unwrap(),
        Reply::Integer(1)
    );
    assert_eq!(
        client.command_str(&["GET", "user:1"]).await.unwrap(),
        Reply::Null
    );
    fake.shutdown().await;
}
#[tokio::test]
async fn ttl_is_set_by_px_and_cleared_by_plain_set() {
    let fake = FakeRedis::start().await;
    let client = Client::from_url(&fake.url()).await.unwrap();
    client
        .command_str(&["SET", "ttl", "v", "PX", "60000"])
        .await
        .unwrap();
    let remaining = client.command_str(&["PTTL", "ttl"]).await.unwrap();
    assert!(remaining.as_integer().unwrap() > 0);
    assert!(fake.state().ttl_ms("ttl").unwrap() > 0);
    client.command_str(&["SET", "ttl", "v2"]).await.unwrap();
    assert_eq!(
        client.command_str(&["PTTL", "ttl"]).await.unwrap(),
        Reply::Integer(-1)
    );
    assert_eq!(fake.state().ttl_ms("ttl"), Some(-1));
    assert_eq!(
        client.command_str(&["TTL", "missing"]).await.unwrap(),
        Reply::Integer(-2)
    );
    fake.shutdown().await;
}
#[tokio::test]
async fn pipeline_preserves_order() {
    let fake = FakeRedis::start().await;
    let client = Client::from_url(&fake.url()).await.unwrap();
    let replies = client
        .pipeline(&[
            vec![b"SET".to_vec(), b"a".to_vec(), b"1".to_vec()],
            vec![b"GET".to_vec(), b"a".to_vec()],
            vec![b"EXISTS".to_vec(), b"a".to_vec()],
        ])
        .await
        .unwrap();
    assert_eq!(
        replies,
        vec![
            Reply::Simple("OK".to_owned()),
            Reply::Bulk(b"1".to_vec()),
            Reply::Integer(1),
        ]
    );
    assert_eq!(client.pipeline(&[]).await.unwrap(), Vec::<Reply>::new());
    assert_eq!(
        client.pipeline(&[Vec::new()]).await.unwrap_err().kind(),
        ErrorKind::Validation
    );
    fake.shutdown().await;
}
#[tokio::test]
async fn sets_and_sorted_sets_round_trip() {
    let fake = FakeRedis::start().await;
    let client = Client::from_url(&fake.url()).await.unwrap();
    assert_eq!(
        client
            .command_str(&["SADD", "tags", "a", "b", "a"])
            .await
            .unwrap(),
        Reply::Integer(2)
    );
    assert_eq!(
        client
            .command_str(&["SISMEMBER", "tags", "b"])
            .await
            .unwrap(),
        Reply::Integer(1)
    );
    assert_eq!(
        client.command_str(&["SMEMBERS", "tags"]).await.unwrap(),
        Reply::Array(vec![Reply::Bulk(b"a".to_vec()), Reply::Bulk(b"b".to_vec()),])
    );
    assert_eq!(
        client
            .command_str(&["ZADD", "rank", "2", "b", "1", "a"])
            .await
            .unwrap(),
        Reply::Integer(2)
    );
    assert_eq!(
        client
            .command_str(&["ZRANGE", "rank", "0", "-1", "WITHSCORES"])
            .await
            .unwrap(),
        Reply::Array(vec![
            Reply::Bulk(b"a".to_vec()),
            Reply::Bulk(b"1".to_vec()),
            Reply::Bulk(b"b".to_vec()),
            Reply::Bulk(b"2".to_vec()),
        ])
    );
    assert_eq!(
        client.command_str(&["ZCARD", "rank"]).await.unwrap(),
        Reply::Integer(2)
    );
    assert_eq!(
        client.command_str(&["TYPE", "rank"]).await.unwrap(),
        Reply::Simple("zset".to_owned())
    );
    assert_eq!(fake.state().set_members("tags"), vec!["a", "b"]);
    assert_eq!(fake.state().zset("rank").len(), 2);
    fake.shutdown().await;
}
#[tokio::test]
async fn auth_succeeds_with_configured_credentials() {
    let fake = FakeRedis::start_with(FakeOptions::auth_required(Some("alice"), "secret")).await;
    let url = format!("redis://alice:secret@{}/0", fake.addr());
    let client = Client::from_url(&url).await.unwrap();
    client.ping().await.unwrap();
    assert_eq!(fake.state().auth_attempts(), 1);
    assert!(fake.state().commands().contains(&"AUTH".to_owned()));
    fake.shutdown().await;
}
#[tokio::test]
async fn auth_failure_is_a_transport_error_without_the_password() {
    let fake = FakeRedis::start_with(FakeOptions::auth_required(Some("alice"), "secret")).await;
    let url = format!("redis://alice:wrong@{}/0", fake.addr());
    let error = Client::from_url(&url).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Transport);
    assert!(error.to_string().contains("WRONGPASS"));
    assert!(!error.to_string().contains("wrong"));
    // The connection is left closed, so the client cannot be built.
    assert_eq!(fake.state().auth_attempts(), 1);
    fake.shutdown().await;
}
#[tokio::test]
async fn wrong_type_replies_are_validation_errors() {
    let fake = FakeRedis::start().await;
    let client = Client::from_url(&fake.url()).await.unwrap();
    client
        .command_str(&["SADD", "aset", "member"])
        .await
        .unwrap();
    let error = client.command_str(&["GET", "aset"]).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("WRONGTYPE"));
    fake.shutdown().await;
}
#[tokio::test]
async fn moved_replies_are_unsupported() {
    let fake = FakeRedis::start_with(FakeOptions::moved_everything()).await;
    let client = Client::from_url(&fake.url()).await.unwrap();
    let error = client.command_str(&["GET", "key"]).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.to_string().contains("MOVED/ASK"));
    fake.shutdown().await;
}
#[tokio::test]
async fn timeouts_name_the_command_but_never_the_arguments() {
    let fake = FakeRedis::start_with(
        FakeOptions::default().with_command_delay(Duration::from_millis(300)),
    )
    .await;
    let mut options = ConnectionOptions::from_url(&format!("redis://{}", fake.addr())).unwrap();
    options.timeout = Duration::from_millis(30);
    let client = Client::connect(options).await.unwrap();
    let error = client
        .command_str(&["GET", "top-secret-key"])
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Timeout);
    let message = error.to_string();
    assert!(message.contains("GET"), "message {message}");
    assert!(!message.contains("top-secret-key"), "message {message}");
    fake.shutdown().await;
}
#[tokio::test]
async fn cluster_url_connects_as_a_single_node() {
    let fake = FakeRedis::start().await;
    let url = format!("redis+cluster://{}/0", fake.addr());
    let client = Client::from_url(&url).await.unwrap();
    assert!(client.options().cluster_mode);
    client.ping().await.unwrap();
    fake.shutdown().await;
}
#[tokio::test]
async fn empty_commands_are_rejected_before_io() {
    let fake = FakeRedis::start().await;
    let client = Client::from_url(&fake.url()).await.unwrap();
    assert_eq!(
        client.command(&[]).await.unwrap_err().kind(),
        ErrorKind::Validation
    );
    fake.shutdown().await;
}
