//! Unit tests for the live `Redis` key-value backend.

// Python counterpart: `tests/unit_tests/extensions/checkpointer/test_redis_checkpointer.py`
// exercises `Redis` through the checkpointer; the Python tree has no dedicated
// `RedisStore` unit test.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::*;
use crate::redis::testsupport::FakeRedis;
use openjiuwen_common::error::ErrorKind;
use serde_json::json;

/// A configuration carrying only `url`.
fn url_config(url: &str) -> RedisStoreConfig {
    RedisStoreConfig {
        url: Some(url.to_string()),
        ..RedisStoreConfig::default()
    }
}

/// A store bound to a running fake server (the client connects lazily).
fn store_for(fake: &FakeRedis) -> RedisStore {
    RedisStore::new(url_config(&fake.url())).unwrap()
}

#[test]
fn every_accepted_scheme_builds_a_store() {
    for scheme in ACCEPTED_URL_SCHEMES {
        let url = format!("{scheme}localhost:6379");
        let store = RedisStore::new(url_config(&url)).unwrap();
        assert_eq!(store.config().url.as_deref(), Some(url.as_str()));
    }
}

#[test]
fn invalid_scheme_is_rejected() {
    let error = RedisStore::new(url_config("http://localhost:6379")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("redis://"));
}

#[test]
fn config_needs_a_url_or_connection_args() {
    let error = RedisStore::new(RedisStoreConfig::default()).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    let config = RedisStoreConfig {
        connection_args: BTreeMap::from([("host".to_string(), json!("localhost"))]),
        ..RedisStoreConfig::default()
    };
    let store = RedisStore::new(config).unwrap();
    assert_eq!(store.config().url, None);
    assert_eq!(store.config().cluster_mode, Some(false));
}

#[test]
fn cluster_mode_is_inferred_from_the_url() {
    let standalone = RedisStore::new(url_config("redis://localhost:6379")).unwrap();
    assert_eq!(standalone.config().cluster_mode, Some(false));
    let tls_standalone = RedisStore::new(url_config("rediss://localhost:6379")).unwrap();
    assert_eq!(tls_standalone.config().cluster_mode, Some(false));
    let cluster = RedisStore::new(url_config("redis+cluster://localhost:6379")).unwrap();
    assert_eq!(cluster.config().cluster_mode, Some(true));
    let tls_cluster = RedisStore::new(url_config("rediss+cluster://localhost:6379")).unwrap();
    assert_eq!(tls_cluster.config().cluster_mode, Some(true));
    let explicit = RedisStore::new(RedisStoreConfig {
        cluster_mode: Some(false),
        ..url_config("redis+cluster://localhost:6379")
    })
    .unwrap();
    assert_eq!(explicit.config().cluster_mode, Some(false));
}

#[test]
fn config_serde_round_trips() {
    let config = RedisStoreConfig {
        url: Some("redis://localhost:6379".to_string()),
        cluster_mode: Some(false),
        connection_args: BTreeMap::from([("db".to_string(), json!(1))]),
    };
    let encoded = serde_json::to_value(&config).unwrap();
    assert_eq!(
        encoded,
        json!({
            "url": "redis://localhost:6379",
            "cluster_mode": false,
            "connection_args": {"db": 1},
        })
    );
    let decoded: RedisStoreConfig = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded, config);
}

#[test]
fn connection_args_override_host_and_port() {
    let store = RedisStore::new(RedisStoreConfig {
        connection_args: BTreeMap::from([
            ("host".to_string(), json!("example.test")),
            ("port".to_string(), json!("6380")),
        ]),
        ..RedisStoreConfig::default()
    })
    .unwrap();
    let options = connection_options(store.config()).unwrap();
    assert_eq!(options.host, "example.test");
    assert_eq!(options.port, 6380);

    for invalid in [json!("not-a-port"), json!(70000), json!(0), json!(true)] {
        let store = RedisStore::new(RedisStoreConfig {
            connection_args: BTreeMap::from([("port".to_string(), invalid.clone())]),
            ..RedisStoreConfig::default()
        })
        .unwrap();
        let error = connection_options(store.config()).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "port {invalid}");
        assert!(error.message().contains("port"), "port {invalid}");
    }

    let store = RedisStore::new(RedisStoreConfig {
        connection_args: BTreeMap::from([("host".to_string(), json!(42))]),
        ..RedisStoreConfig::default()
    })
    .unwrap();
    let error = connection_options(store.config()).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("host"));
}

#[test]
fn debug_redacts_credentials() {
    let store = RedisStore::new(RedisStoreConfig {
        url: Some("redis://user:secret@localhost:6379".to_string()),
        connection_args: BTreeMap::from([
            ("password".to_string(), json!("hunter2")),
            ("db".to_string(), json!(1)),
        ]),
        ..RedisStoreConfig::default()
    })
    .unwrap();
    let rendered = format!("{store:?}");
    assert!(!rendered.contains("secret"), "{rendered}");
    assert!(!rendered.contains("hunter2"), "{rendered}");
    assert!(rendered.contains("redis://localhost:6379"), "{rendered}");
    // Key names remain visible for debugging; only the values are hidden.
    assert!(rendered.contains("password"), "{rendered}");
    assert!(rendered.contains("db"), "{rendered}");
}

#[test]
fn glob_metacharacters_are_escaped() {
    assert_eq!(escape_glob("plain/"), "plain/");
    assert_eq!(escape_glob("a*b?c[d]e\\f"), "a\\*b\\?c\\[d\\]e\\\\f");
}

#[test]
fn sub_millisecond_expiries_round_up() {
    assert_eq!(expiry_millis(Duration::ZERO), 1);
    assert_eq!(expiry_millis(Duration::from_nanos(1)), 1);
    assert_eq!(expiry_millis(Duration::from_micros(999)), 1);
    assert_eq!(expiry_millis(Duration::from_millis(250)), 250);
    assert_eq!(expiry_millis(Duration::from_secs(2)), 2_000);
}

#[test]
fn damaged_payloads_are_rejected_by_key() {
    let empty = decode_value("key", &[]).unwrap_err();
    assert_eq!(empty.kind(), ErrorKind::Validation);
    let message = empty.message().to_string();
    assert!(message.contains("key"), "{message}");

    let unknown = decode_value("key", &[0x7f, b'v']).unwrap_err();
    assert_eq!(unknown.kind(), ErrorKind::Validation);
    let message = unknown.message().to_string();
    assert!(message.contains("0x7f"), "{message}");

    let not_utf8 = decode_value("key", &[TEXT_TAG, 0xff]).unwrap_err();
    assert_eq!(not_utf8.kind(), ErrorKind::Validation);
    let message = not_utf8.message().to_string();
    assert!(message.contains("UTF-8"), "{message}");
}

#[test]
fn unknown_unlink_is_detected_from_the_server_message() {
    let error = Error::transport(
        "redis server error: ERR unknown command 'UNLINK', with args beginning with: key",
    );
    assert!(is_unknown_command(&error, "UNLINK"));
    assert!(!is_unknown_command(&error, "DEL"));
    let moved = Error::transport("redis server error: MOVED 3999 127.0.0.1:7000");
    assert!(!is_unknown_command(&moved, "UNLINK"));
}

#[tokio::test]
async fn text_values_round_trip() {
    let fake = FakeRedis::start().await;
    let store = store_for(&fake);
    store.set("greeting", KvValue::text("héllo")).await.unwrap();
    assert_eq!(
        store.get("greeting").await.unwrap(),
        Some(KvValue::text("héllo"))
    );
    assert!(store.exists("greeting").await.unwrap());
    assert_eq!(store.get("missing").await.unwrap(), None);
    assert!(!store.exists("missing").await.unwrap());
    fake.shutdown().await;
}

#[tokio::test]
async fn binary_values_keep_their_kind() {
    let fake = FakeRedis::start().await;
    let store = store_for(&fake);
    store
        .set("blob", KvValue::bytes(vec![0, 1, 2, 255]))
        .await
        .unwrap();
    assert_eq!(
        store.get("blob").await.unwrap(),
        Some(KvValue::bytes(vec![0, 1, 2, 255]))
    );
    // A payload that is valid UTF-8 but written as bytes must stay `Bytes`.
    store
        .set("ascii", KvValue::bytes(b"plain".to_vec()))
        .await
        .unwrap();
    assert_eq!(
        store.get("ascii").await.unwrap(),
        Some(KvValue::bytes(b"plain".to_vec()))
    );
    // The stored shape is the bytes tag followed by the payload.
    assert_eq!(
        fake.state().get("ascii"),
        Some(vec![BYTES_TAG, b'p', b'l', b'a', b'i', b'n'])
    );
    fake.shutdown().await;
}

#[tokio::test]
async fn delete_removes_the_key() {
    let fake = FakeRedis::start().await;
    let store = store_for(&fake);
    store.set("gone", KvValue::text("value")).await.unwrap();
    store.delete("gone").await.unwrap();
    assert!(!store.exists("gone").await.unwrap());
    // Deleting a missing key is a no-op.
    store.delete("gone").await.unwrap();
    fake.shutdown().await;
}

#[tokio::test]
async fn exclusive_set_writes_once_and_honours_the_ttl() {
    let fake = FakeRedis::start().await;
    let store = store_for(&fake);
    assert!(store
        .exclusive_set("lock", KvValue::text("first"), None)
        .await
        .unwrap());
    assert!(!store
        .exclusive_set(
            "lock",
            KvValue::text("second"),
            Some(Duration::from_secs(60))
        )
        .await
        .unwrap());
    assert_eq!(
        store.get("lock").await.unwrap(),
        Some(KvValue::text("first"))
    );
    assert_eq!(fake.state().ttl_ms("lock"), Some(-1), "no expiry was set");

    assert!(store
        .exclusive_set(
            "expiring",
            KvValue::text("value"),
            Some(Duration::from_secs(60))
        )
        .await
        .unwrap());
    let ttl = fake
        .state()
        .ttl_ms("expiring")
        .expect("the key is live and carries a TTL");
    assert!(ttl > 0, "a positive TTL must be stored: {ttl}");

    // A plain `set` clears the TTL, as the kernel contract requires.
    store
        .set("expiring", KvValue::text("replaced"))
        .await
        .unwrap();
    assert_eq!(
        fake.state().ttl_ms("expiring"),
        Some(-1),
        "SET clears the TTL"
    );
    fake.shutdown().await;
}

#[tokio::test]
async fn mget_preserves_input_order() {
    let fake = FakeRedis::start().await;
    let store = store_for(&fake);
    store.set("a", KvValue::text("1")).await.unwrap();
    store.set("b", KvValue::bytes(vec![2])).await.unwrap();
    let keys = vec!["a".to_string(), "missing".to_string(), "b".to_string()];
    assert_eq!(
        store.mget(&keys).await.unwrap(),
        vec![
            Some(KvValue::text("1")),
            None,
            Some(KvValue::bytes(vec![2]))
        ]
    );
    assert_eq!(store.mget(&[]).await.unwrap(), Vec::new());
    fake.shutdown().await;
}

#[tokio::test]
async fn prefix_reads_and_deletes_count_live_keys() {
    let fake = FakeRedis::start().await;
    let store = store_for(&fake);
    for (key, value) in [("pre/a", "a"), ("pre/b", "b"), ("other/c", "c")] {
        store.set(key, KvValue::text(value)).await.unwrap();
    }
    let entries = store.get_by_prefix("pre/").await.unwrap();
    assert_eq!(entries.keys().collect::<Vec<_>>(), vec!["pre/a", "pre/b"]);
    assert_eq!(entries.get("pre/b"), Some(&KvValue::text("b")));
    assert_eq!(store.get_by_prefix("").await.unwrap().len(), 3);

    // The prefix metacharacters are escaped before they reach `SCAN MATCH`.
    // The fake matcher has no backslash support, so an escaped `*` matches
    // nothing there while an unescaped `st*r/` would match both keys; the
    // assertion proves the pattern carries the escape.
    store.set("star/a", KvValue::text("a")).await.unwrap();
    store.set("stXr/a", KvValue::text("x")).await.unwrap();
    assert!(store.get_by_prefix("st*r/").await.unwrap().is_empty());

    assert_eq!(store.delete_by_prefix("pre/").await.unwrap(), 2);
    assert_eq!(store.delete_by_prefix("pre/").await.unwrap(), 0);
    assert!(store.get_by_prefix("pre/").await.unwrap().is_empty());
    fake.shutdown().await;
}

#[tokio::test]
async fn batch_delete_counts_live_keys_once() {
    let fake = FakeRedis::start().await;
    let store = store_for(&fake);
    store.set("a", KvValue::text("a")).await.unwrap();
    store.set("b", KvValue::text("b")).await.unwrap();
    let keys = vec![
        "a".to_string(),
        "missing".to_string(),
        "b".to_string(),
        "a".to_string(),
    ];
    assert_eq!(store.batch_delete(&keys).await.unwrap(), 2);
    assert_eq!(store.batch_delete(&[]).await.unwrap(), 0);
    assert!(!store.exists("a").await.unwrap());
    fake.shutdown().await;
}

#[tokio::test]
async fn pipeline_executes_in_order_and_clears_its_queue() {
    let fake = FakeRedis::start().await;
    let store = store_for(&fake);
    let mut pipeline = store.pipeline();
    pipeline.set("alpha", KvValue::text("one"));
    pipeline.get("alpha");
    pipeline.exists("alpha");
    pipeline.exists("missing");
    assert_eq!(
        pipeline.execute().await.unwrap(),
        vec![
            None,
            Some(KvValue::text("one")),
            Some(KvValue::text("true")),
            Some(KvValue::text("false")),
        ]
    );
    // The batch is sent in order, so the queued `get` sees the queued `set`.
    // The handshake commands (`SELECT` for the URL's `/0` database) come first.
    let commands: Vec<String> = fake
        .state()
        .commands()
        .into_iter()
        .filter(|command| !matches!(command.as_str(), "AUTH" | "SELECT" | "HELLO"))
        .collect();
    assert_eq!(commands, vec!["SET", "GET", "EXISTS", "EXISTS"]);
    // Reusing the pipeline must not replay the executed operations.
    assert_eq!(pipeline.execute().await.unwrap(), Vec::new());
    fake.shutdown().await;
}

#[tokio::test]
async fn connect_fails_fast_for_a_closed_port() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    let error = RedisStore::connect(url_config(&format!("redis://{addr}/0")))
        .await
        .err()
        .expect("a closed port must fail the eager connect");
    assert_eq!(error.kind(), ErrorKind::Transport);
}

#[tokio::test]
async fn connect_uses_tls_and_reports_a_failed_handshake() {
    // `rediss://` now performs a real TLS handshake; against a plain RESP listener it
    // fails as a transport error naming TLS rather than as an unsupported feature.
    let fake = FakeRedis::start().await;
    let error = RedisStore::connect(url_config(&format!("rediss://{}/0", fake.addr())))
        .await
        .err()
        .expect("a plain listener cannot complete a TLS handshake");
    assert_eq!(error.kind(), ErrorKind::Transport);
    let message = error.message().to_string();
    assert!(message.contains("TLS"), "{message}");
    fake.shutdown().await;
}

#[tokio::test]
async fn cluster_mode_against_a_standalone_server_is_rejected() {
    let fake = FakeRedis::start().await;
    let error = RedisStore::connect(RedisStoreConfig {
        url: Some(fake.url()),
        cluster_mode: Some(true),
        ..RedisStoreConfig::default()
    })
    .await
    .err()
    .expect("a standalone node must not be accepted in cluster mode");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.message().to_string();
    assert!(message.contains("cluster_enabled:1"), "{message}");
    fake.shutdown().await;
}
