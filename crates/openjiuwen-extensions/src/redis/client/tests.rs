#![allow(clippy::unwrap_used, clippy::expect_used)]
use super::cluster::{self, NodeAddr, RedirectKind};
use super::*;
use crate::redis::testsupport::{FakeOptions, FakeRedis};
use openjiuwen_common::error::ErrorKind;
use serde_json::json;
use std::net::SocketAddr;
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
    assert!(!options.tls);
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
fn tls_schemes_record_the_tls_flag() {
    let tls = ConnectionOptions::from_url("rediss://example.com:6380/2").unwrap();
    assert!(tls.tls, "rediss:// selects TLS");
    assert!(!tls.cluster_mode);
    assert_eq!(tls.port, 6380);
    assert_eq!(tls.db, Some(2));
    let cluster = ConnectionOptions::from_url("rediss+cluster://example.com:7000").unwrap();
    assert!(cluster.tls, "rediss+cluster:// selects TLS");
    assert!(
        cluster.cluster_mode,
        "rediss+cluster:// selects cluster mode"
    );
    let plain = ConnectionOptions::from_url("redis://example.com").unwrap();
    assert!(!plain.tls);
    assert!(!plain.cluster_mode);
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
async fn moved_is_followed_once_and_caches_the_slot() {
    // `12739` is the hash slot of the key `123456789`.
    let owner = FakeRedis::start().await;
    let owner_client = Client::from_url(&owner.url()).await.unwrap();
    owner_client
        .command_str(&["SET", "123456789", "ada"])
        .await
        .unwrap();
    let seed = FakeRedis::start_with(FakeOptions::moved_to(12739, owner.addr())).await;
    let client = Client::from_url(&seed.url()).await.unwrap();
    assert_eq!(
        client.command_str(&["GET", "123456789"]).await.unwrap(),
        Reply::Bulk(b"ada".to_vec())
    );
    assert_eq!(seed.state().commands(), ["SELECT", "GET"]);
    // The owner sees the test client's own `SELECT`/`SET` plus the redirected
    // connection's handshake and `GET`.
    assert_eq!(owner.state().commands(), ["SELECT", "SET", "SELECT", "GET"]);
    // The slot is cached, so an equivalent hash tag goes straight to the owner node.
    assert_eq!(
        client
            .command_str(&["GET", "{123456789}other"])
            .await
            .unwrap(),
        Reply::Null
    );
    assert_eq!(
        seed.state().commands(),
        ["SELECT", "GET"],
        "the cached slot must not go back to the seed"
    );
    assert_eq!(
        owner.state().commands(),
        ["SELECT", "SET", "SELECT", "GET", "GET"]
    );
    seed.shutdown().await;
    owner.shutdown().await;
}
#[tokio::test]
async fn a_second_consecutive_moved_is_a_transport_error() {
    let dead = closed_address().await;
    let hop = FakeRedis::start_with(FakeOptions::moved_to(3999, dead)).await;
    let seed = FakeRedis::start_with(FakeOptions::moved_to(3999, hop.addr())).await;
    let client = Client::from_url(&seed.url()).await.unwrap();
    let error = client.command_str(&["GET", "key"]).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Transport);
    let message = error.to_string();
    assert!(
        message.contains("second consecutive redirection"),
        "{message}"
    );
    assert!(message.contains("MOVED 3999"), "{message}");
    // The redirection was followed exactly once: one `MOVED` command, no loop.
    assert_eq!(seed.state().commands(), ["SELECT", "GET"]);
    assert_eq!(hop.state().commands(), ["SELECT", "GET"]);
    seed.shutdown().await;
    hop.shutdown().await;
}
#[tokio::test]
async fn ask_sends_asking_before_the_command_and_is_not_cached() {
    let owner = FakeRedis::start().await;
    let seed = FakeRedis::start_with(FakeOptions::ask_to(3999, owner.addr())).await;
    let client = Client::from_url(&seed.url()).await.unwrap();
    assert_eq!(
        client.command_str(&["GET", "k"]).await.unwrap(),
        Reply::Null
    );
    assert_eq!(
        owner.state().first_connection_commands(),
        ["SELECT", "ASKING", "GET"],
        "ASKING must precede the redirected command on the same connection"
    );
    assert_eq!(owner.state().connection_count(), 1);
    // `ASK` is a one-shot redirection, so the next command starts at the seed again.
    assert_eq!(
        client.command_str(&["GET", "k"]).await.unwrap(),
        Reply::Null
    );
    assert_eq!(seed.state().commands(), ["SELECT", "GET", "GET"]);
    assert_eq!(
        owner.state().commands(),
        ["SELECT", "ASKING", "GET", "ASKING", "GET"]
    );
    seed.shutdown().await;
    owner.shutdown().await;
}
#[tokio::test]
async fn a_pipeline_follows_redirects_and_keeps_positions() {
    let owner = FakeRedis::start().await;
    let seed = FakeRedis::start_with(FakeOptions::moved_to(3999, owner.addr())).await;
    let client = Client::from_url(&seed.url()).await.unwrap();
    let replies = client
        .pipeline(&[
            vec![b"SET".to_vec(), b"a".to_vec(), b"1".to_vec()],
            vec![b"GET".to_vec(), b"missing".to_vec()],
            vec![b"EXISTS".to_vec(), b"a".to_vec()],
        ])
        .await
        .unwrap();
    assert_eq!(
        replies,
        vec![
            Reply::Simple("OK".to_owned()),
            Reply::Null,
            Reply::Integer(1),
        ]
    );
    assert_eq!(seed.state().commands(), ["SELECT", "SET", "GET", "EXISTS"]);
    assert_eq!(owner.state().commands(), ["SELECT", "SET", "GET", "EXISTS"]);
    seed.shutdown().await;
    owner.shutdown().await;
}
#[tokio::test]
async fn cluster_mode_against_a_standalone_node_is_rejected() {
    let fake = FakeRedis::start().await;
    let url = format!("redis+cluster://{}/0", fake.addr());
    let error = Client::from_url(&url).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("cluster_enabled:1"), "{message}");
    assert!(message.contains("not a cluster node"), "{message}");
    fake.shutdown().await;
}
#[tokio::test]
async fn cluster_mode_against_a_cluster_node_connects() {
    let fake = FakeRedis::start_with(FakeOptions::cluster()).await;
    let url = format!("redis+cluster://{}/0", fake.addr());
    let client = Client::from_url(&url).await.unwrap();
    assert!(client.options().cluster_mode);
    client.ping().await.unwrap();
    // The node was verified and its slot map read before the first command.
    assert_eq!(
        fake.state().commands(),
        ["SELECT", "CLUSTER", "CLUSTER", "PING"]
    );
    fake.shutdown().await;
}
#[tokio::test]
async fn cluster_mode_presees_slots_from_cluster_slots() {
    let owner = FakeRedis::start().await;
    let owner_client = Client::from_url(&owner.url()).await.unwrap();
    owner_client.command_str(&["SET", "k", "v"]).await.unwrap();
    let seed = FakeRedis::start_with(FakeOptions::cluster_owned_by(
        "127.0.0.1",
        owner.addr().port(),
    ))
    .await;
    let url = format!("redis+cluster://{}/0", seed.addr());
    let client = Client::from_url(&url).await.unwrap();
    assert_eq!(
        client.command_str(&["GET", "k"]).await.unwrap(),
        Reply::Bulk(b"v".to_vec())
    );
    assert_eq!(seed.state().commands(), ["SELECT", "CLUSTER", "CLUSTER"]);
    assert_eq!(owner.state().commands(), ["SELECT", "SET", "SELECT", "GET"]);
    seed.shutdown().await;
    owner.shutdown().await;
}
#[tokio::test]
async fn a_rediss_url_against_a_plain_listener_fails_the_handshake() {
    // No certificate fixture is available offline, so a successful handshake is not
    // exercised here; what matters is that TLS is attempted (never `Unsupported`)
    // and that a peer that does not speak TLS surfaces as a transport error.
    let fake = FakeRedis::start().await;
    let url = format!("rediss://{}/0", fake.addr());
    let error = Client::from_url(&url).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Transport);
    let message = error.to_string();
    assert!(message.contains("TLS"), "{message}");
    assert!(!message.contains("Unsupported"), "{message}");
    fake.shutdown().await;
}
#[test]
fn redirection_messages_parse_slots_and_nodes() {
    let moved = cluster::parse_redirect("MOVED 3999 127.0.0.1:7000").unwrap();
    assert_eq!(moved.kind, RedirectKind::Moved);
    assert_eq!(moved.slot, 3999);
    assert_eq!(moved.addr, NodeAddr::new("127.0.0.1", 7000));
    assert_eq!(moved.raw, "MOVED 3999 127.0.0.1:7000");
    let ask = cluster::parse_redirect("ASK 12 [::1]:7001").unwrap();
    assert_eq!(ask.kind, RedirectKind::Ask);
    assert_eq!(ask.slot, 12);
    assert_eq!(ask.addr, NodeAddr::new("::1", 7001));
    for other in [
        "ERR unknown command",
        "WRONGTYPE Operation against a key",
        "MOVED not-a-slot 127.0.0.1:7000",
        "MOVED 1 missing-port",
        "MOVED 1 host:0",
        "MOVED 1 :7000",
        "MOVED 70000 127.0.0.1:7000",
    ] {
        assert!(cluster::parse_redirect(other).is_none(), "message {other}");
    }
}
#[test]
fn hash_slots_match_redis_and_honour_hash_tags() {
    // Reference values from `CLUSTER KEYSLOT`.
    assert_eq!(cluster::hash_slot(b"123456789"), 12_739);
    assert_eq!(cluster::hash_slot(b"foo"), 12_182);
    assert_eq!(cluster::hash_slot(b"bar"), 5_061);
    // A `{tag}` restricts hashing to the tagged substring.
    assert_eq!(cluster::hash_slot(b"{user1000}.following"), 3_443);
    assert_eq!(cluster::hash_slot(b"{user1000}.followers"), 3_443);
    // An empty tag and an unterminated tag both hash the whole key.
    assert_eq!(cluster::hash_slot(b"{}foo"), 9_500);
    assert_ne!(cluster::hash_slot(b"{}foo"), cluster::hash_slot(b"foo"));
    assert_eq!(cluster::hash_slot(b"{unclosed"), 470);
    assert_ne!(
        cluster::hash_slot(b"{user1000}.following"),
        cluster::hash_slot(b"{user1001}.following")
    );
}
#[test]
fn only_single_key_commands_consult_the_slot_cache() {
    assert_eq!(cluster::command_slot(&[b"GET", b"123456789"]), Some(12_739));
    assert_eq!(
        cluster::command_slot(&[b"get", b"123456789"]),
        Some(12_739),
        "command names are matched case-insensitively"
    );
    // Keyless, multi-key, and unknown commands always start at the seed node.
    assert_eq!(cluster::command_slot(&[b"PING"]), None);
    assert_eq!(cluster::command_slot(&[b"MGET", b"123456789"]), None);
    assert_eq!(
        cluster::command_slot(&[b"SCAN", b"0", b"MATCH", b"*"]),
        None
    );
    assert_eq!(cluster::command_slot(&[]), None);
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
/// A loopback address with nothing listening on it.
async fn closed_address() -> SocketAddr {
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .expect("loopback listener binds");
    let addr = listener
        .local_addr()
        .expect("bound listener has an address");
    drop(listener);
    addr
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

/// Dropping a command future in flight must never leave the abandoned reply on
/// the wire for the next command to read.
///
/// `RESP` replies carry no request id, so a caller-side `timeout`/`select!`/
/// abort drops `exchange` between the write and the read. Without an RAII guard
/// the socket stays open with the abandoned reply behind it, and the next
/// command reads that reply instead of its own — silent wrong data on a
/// checkpoint or store read.
#[tokio::test]
async fn a_dropped_command_future_never_hands_its_reply_to_the_next_command() {
    let fake = FakeRedis::start_with(
        FakeOptions::default().with_command_delay(Duration::from_millis(150)),
    )
    .await;
    let client = Client::from_url(&fake.url()).await.unwrap();
    client
        .command_str(&["SET", "first", "alpha"])
        .await
        .unwrap();
    client
        .command_str(&["SET", "second", "beta"])
        .await
        .unwrap();
    let connections_before = fake.state().connection_count();

    // Abort the command with a caller-side timeout while the fake is still
    // holding the reply back.
    let dropped = tokio::time::timeout(
        Duration::from_millis(50),
        client.command_str(&["GET", "first"]),
    )
    .await;
    assert!(
        dropped.is_err(),
        "the caller-side timeout drops the in-flight command"
    );
    // Let the fake finish the abandoned command, so its reply really is on the
    // wire (in the closed socket's buffer, or lost with it).
    tokio::time::sleep(Duration::from_millis(200)).await;

    let reply = client
        .command_str(&["GET", "second"])
        .await
        .expect("the next command succeeds");
    assert_eq!(
        reply,
        Reply::Bulk(b"beta".to_vec()),
        "the dropped command's reply must never be reused"
    );
    assert!(
        fake.state().connection_count() > connections_before,
        "the dropped command closes the connection so the next command reconnects"
    );
    fake.shutdown().await;
}

/// A loopback peer that answers every accepted connection with canned bytes and
/// then closes it, in acceptance order.
///
/// Writing on accept (before the client's command even arrives) keeps the test
/// deterministic: every byte of the canned reply is already queued when the
/// client reads.
async fn canned_reply_peer(answers: Vec<&'static [u8]>) -> SocketAddr {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::AsyncWriteExt as _;

    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .expect("loopback listener binds");
    let addr = listener.local_addr().expect("listener has an address");
    tokio::spawn(async move {
        let accepted = AtomicUsize::new(0);
        while let Ok((mut stream, _)) = listener.accept().await {
            let index = accepted.fetch_add(1, Ordering::SeqCst);
            if let Some(answer) = answers.get(index) {
                let _ = stream.write_all(answer).await;
            }
        }
    });
    addr
}

/// Bytes the peer sends beyond the reply the client asked for cannot be
/// attributed to any command: the connection is closed and the command fails
/// with a typed error instead of reusing the wire.
#[tokio::test]
async fn buffered_unsolicited_replies_are_a_typed_desynchronisation_error() {
    let addr = canned_reply_peer(vec![b"+FIRST\r\n+UNSOLICITED\r\n", b"+THIRD\r\n"]).await;
    let client = Client::from_url(&format!("redis://{addr}")).await.unwrap();
    assert_eq!(
        client.command_str(&["GET", "a"]).await.unwrap(),
        Reply::Simple("FIRST".to_string())
    );
    let error = client
        .command_str(&["GET", "b"])
        .await
        .expect_err("the connection is desynchronised");
    assert_eq!(error.kind(), ErrorKind::Transport);
    assert!(
        error.to_string().contains("desynchronised"),
        "the error names the cause: {error}"
    );
    assert_eq!(
        client.command_str(&["GET", "c"]).await.unwrap(),
        Reply::Simple("THIRD".to_string()),
        "the next command reconnects instead of reusing the wire"
    );
}
