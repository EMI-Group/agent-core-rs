#![cfg(feature = "redis")]
#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Crate-boundary integration suite for the Redis adapters.
//!
//! Every assertion enters through a public path (`openjiuwen_extensions::redis`,
//! `store::redis`, `store::build_kv_store`, `checkpointer::redis`), so a broken
//! re-export, a changed factory signature, or a lost deferral contract fails here
//! before a consumer notices. The peer these tests talk to is the [`fake`] server
//! below: it binds `127.0.0.1:0`, keeps its data in memory, and answers `RESP2`,
//! so the suite is local, offline, and deterministic.
//!
//! One section per surface: codec, store, checkpointer, TTL, factory/deferrals.
//! Without the `redis` feature this file compiles to an empty test binary.
use fake::Server;
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_extensions::checkpointer::redis::{
    RedisCheckpointer, RedisCheckpointerConfig, RedisConnectionConfig, RedisTTLConfig,
};
use openjiuwen_extensions::checkpointer::{Checkpoint, CheckpointNamespace, Checkpointer};
use openjiuwen_extensions::redis::codec::{encode_command, parse_reply, Reply};
use openjiuwen_extensions::redis::Client;
use openjiuwen_extensions::store::redis::{RedisStore, RedisStoreConfig};
use openjiuwen_extensions::store::{
    build_document_store, build_kv_store, build_vector_store, KvStore, KvValue, StoreBackend,
};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;
/// Pinned checkpoint timestamps, so list ordering is deterministic.
const T1: &str = "2024-01-01T00:00:00Z";
const T2: &str = "2024-01-02T00:00:00Z";
const T3: &str = "2024-01-03T00:00:00Z";
fn kv(text: &str) -> KvValue {
    KvValue::text(text)
}
/// Unwraps the error of a fallible call whose success type is not `Debug`
/// (several factories return trait objects), failing the test on `Ok`.
fn expect_error<T>(result: Result<T>) -> Error {
    result
        .err()
        .expect("expected an error, got a successful value")
}
/// Parses one complete `RESP` fixture and asserts it was consumed whole.
fn parse_one(input: &[u8]) -> Reply {
    match parse_reply(input).unwrap() {
        Some((reply, consumed)) => {
            assert_eq!(consumed, input.len(), "the fixture must be one reply");
            reply
        }
        None => panic!("incomplete RESP fixture: {input:?}"),
    }
}
/// Opens a `RedisStore` against `fake`, returning the kernel trait object.
async fn connect_store(fake: &Server) -> Arc<dyn KvStore> {
    let config = RedisStoreConfig {
        url: Some(fake.url()),
        ..RedisStoreConfig::default()
    };
    RedisStore::connect(config).await.unwrap()
}
/// Opens a `RedisCheckpointer` against `fake`.
async fn connect_checkpointer(fake: &Server) -> Arc<dyn Checkpointer> {
    let config = RedisCheckpointerConfig::new(RedisConnectionConfig::new(fake.url()));
    RedisCheckpointer::connect(config).await.unwrap()
}
/// Saves a checkpoint whose `created_at` is pinned.
async fn save_at(cp: &dyn Checkpointer, session: &str, at: &str, note: &str) -> String {
    let mut checkpoint = Checkpoint::new(session, CheckpointNamespace::Agent, "agent-1")
        .with_global_state(json!({ "note": note }));
    checkpoint.created_at = at.to_owned();
    cp.save(checkpoint).await.unwrap()
}
/// A `redis://` URL whose port was bound and released, so connecting is refused.
async fn closed_port_url() -> String {
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    format!("redis://{addr}/0")
}
// ---------------------------------------------------------------------------
// Codec
// ---------------------------------------------------------------------------
#[test]
fn encode_command_writes_binary_safe_bulk_strings() {
    assert_eq!(encode_command(&[b"PING"]), b"*1\r\n$4\r\nPING\r\n".to_vec());
    assert_eq!(encode_command(&[]), b"*0\r\n".to_vec());
    let binary: &[u8] = b"a\r\nb\0c\xff";
    let encoded = encode_command(&[b"SET", b"", binary]);
    let mut expected = format!("*3\r\n$3\r\nSET\r\n$0\r\n\r\n${}\r\n", binary.len()).into_bytes();
    expected.extend_from_slice(binary);
    expected.extend_from_slice(b"\r\n");
    assert_eq!(encoded, expected);
    // The encoding is valid RESP: it parses back into the same arguments.
    let expected = Reply::Array(vec![
        Reply::Bulk(b"SET".to_vec()),
        Reply::Bulk(Vec::new()),
        Reply::Bulk(binary.to_vec()),
    ]);
    assert_eq!(parse_one(&encoded), expected);
}
#[test]
fn parse_reply_decodes_resp2_and_resp3_scalars() {
    let cases: [(&[u8], Reply); 10] = [
        (b"+OK\r\n", Reply::Simple("OK".to_owned())),
        (b"-ERR bad\r\n", Reply::Error("ERR bad".to_owned())),
        (b":42\r\n", Reply::Integer(42)),
        (b"$3\r\nabc\r\n", Reply::Bulk(b"abc".to_vec())),
        (b"$0\r\n\r\n", Reply::Bulk(Vec::new())),
        (b"$-1\r\n", Reply::Null),
        (b"_\r\n", Reply::Null),
        (b"#t\r\n", Reply::Boolean(true)),
        (b"#f\r\n", Reply::Boolean(false)),
        (b",3.5\r\n", Reply::Double(3.5)),
    ];
    for (input, expected) in cases {
        assert_eq!(parse_one(input), expected, "input {input:?}");
    }
}
#[test]
fn parse_reply_decodes_aggregated_replies() {
    let simple = |text: &str| Reply::Simple(text.to_owned());
    let cases: [(&[u8], Reply); 5] = [
        (
            b"*2\r\n$1\r\na\r\n:1\r\n",
            Reply::Array(vec![Reply::Bulk(b"a".to_vec()), Reply::Integer(1)]),
        ),
        (
            b"*2\r\n*1\r\n+x\r\n*0\r\n",
            Reply::Array(vec![
                Reply::Array(vec![simple("x")]),
                Reply::Array(Vec::new()),
            ]),
        ),
        (b"*-1\r\n", Reply::Null),
        (
            b"%1\r\n+a\r\n+b\r\n",
            Reply::Map(vec![(simple("a"), simple("b"))]),
        ),
        (b"~1\r\n+x\r\n", Reply::Set(vec![simple("x")])),
    ];
    for (input, expected) in cases {
        assert_eq!(parse_one(input), expected, "input {input:?}");
    }
}
#[test]
fn parse_reply_reports_truncated_input_as_incomplete() {
    let truncated: [&[u8]; 6] = [
        b"",
        b"+",
        b"+OK\r",
        b"$3\r\nab",
        b"*2\r\n$1\r\na\r\n",
        b"$-1\r",
    ];
    for input in truncated {
        assert!(
            parse_reply(input).unwrap().is_none(),
            "expected `Ok(None)` for {input:?}"
        );
    }
    // A complete reply followed by trailing bytes consumes only its own bytes.
    let (reply, consumed) = parse_reply(b"+OK\r\n:7\r\n").unwrap().unwrap();
    assert_eq!((reply, consumed), (Reply::Simple("OK".to_owned()), 5));
}
// ---------------------------------------------------------------------------
// RedisStore (kernel `KvStore` contract)
// ---------------------------------------------------------------------------
#[tokio::test]
async fn store_round_trips_text_and_binary_values() {
    let fake = Server::start().await;
    let store = connect_store(&fake).await;
    store.set("text-key", kv("hello")).await.unwrap();
    assert_eq!(store.get("text-key").await.unwrap(), Some(kv("hello")));
    let raw = fake.state().get("text-key").expect("the payload is stored");
    assert_eq!(raw.first(), Some(&0x00), "text carries the 0x00 kind tag");
    let opaque = vec![0u8, 1, 2, 255];
    store
        .set("bytes-key", KvValue::bytes(opaque.clone()))
        .await
        .unwrap();
    assert_eq!(
        store.get("bytes-key").await.unwrap(),
        Some(KvValue::bytes(opaque))
    );
    // ASCII bytes must not come back as text: the variant survives the round trip.
    store
        .set("ascii", KvValue::bytes(b"plain-ascii-bytes".to_vec()))
        .await
        .unwrap();
    let loaded = store.get("ascii").await.unwrap().unwrap();
    assert!(
        !loaded.is_text(),
        "a byte value decoded as text: {loaded:?}"
    );
    assert_eq!(loaded, KvValue::bytes(b"plain-ascii-bytes".to_vec()));
}
#[tokio::test]
async fn store_missing_keys_read_none_and_delete_is_idempotent() {
    let fake = Server::start().await;
    let store = connect_store(&fake).await;
    assert_eq!(store.get("absent").await.unwrap(), None);
    assert!(!store.exists("absent").await.unwrap());
    store.delete("absent").await.unwrap();
    store.set("present", kv("v")).await.unwrap();
    assert!(store.exists("present").await.unwrap());
    store.delete("present").await.unwrap();
    store.delete("present").await.unwrap();
    assert!(!store.exists("present").await.unwrap());
    assert_eq!(store.get("present").await.unwrap(), None);
}
#[tokio::test]
async fn store_exclusive_set_is_create_only_and_plain_set_clears_the_ttl() {
    let fake = Server::start().await;
    let store = connect_store(&fake).await;
    assert!(store
        .exclusive_set("lock", kv("first"), None)
        .await
        .unwrap());
    assert!(!store
        .exclusive_set("lock", kv("second"), None)
        .await
        .unwrap());
    assert_eq!(store.get("lock").await.unwrap(), Some(kv("first")));
    assert_eq!(fake.state().ttl_ms("lock"), Some(-1), "no expiry was asked");
    let lease = Duration::from_millis(500);
    assert!(store
        .exclusive_set("leased", kv("v"), Some(lease))
        .await
        .unwrap());
    let remaining = fake.state().ttl_ms("leased").expect("the lease exists");
    assert!(
        (1..=500).contains(&remaining),
        "unexpected PTTL {remaining}"
    );
    // A plain `SET` overwrites the value and drops the expiry (kernel contract).
    store.set("leased", kv("reset")).await.unwrap();
    assert_eq!(store.get("leased").await.unwrap(), Some(kv("reset")));
    assert_eq!(fake.state().ttl_ms("leased"), Some(-1));
}
#[tokio::test]
async fn store_mget_preserves_input_order_with_misses() {
    let fake = Server::start().await;
    let store = connect_store(&fake).await;
    store.set("m:a", kv("A")).await.unwrap();
    store.set("m:b", KvValue::bytes(vec![9, 8])).await.unwrap();
    let keys = vec!["m:a".to_owned(), "m:missing".to_owned(), "m:b".to_owned()];
    let values = store.mget(&keys).await.unwrap();
    assert_eq!(values.len(), 3);
    assert_eq!(values[0], Some(kv("A")));
    assert_eq!(values[1], None);
    assert_eq!(values[2], Some(KvValue::bytes(vec![9, 8])));
    assert_eq!(store.mget(&[]).await.unwrap(), Vec::new());
}
#[tokio::test]
async fn store_prefix_scans_escape_globs_and_count_live_keys() {
    let fake = Server::start().await;
    let store = connect_store(&fake).await;
    for index in 0..3 {
        store
            .set(&format!("job:{index}"), kv(&format!("v{index}")))
            .await
            .unwrap();
    }
    store.set("job-other", kv("nope")).await.unwrap();
    let entries = store.get_by_prefix("job:").await.unwrap();
    assert_eq!(entries.len(), 3, "unexpected entries: {entries:?}");
    assert_eq!(entries.get("job:1"), Some(&kv("v1")));
    // A literal `*` in the prefix is escaped, so it cannot act as a wildcard.
    store.set("glob*:a", kv("star")).await.unwrap();
    store.set("globX:a", kv("decoy")).await.unwrap();
    let starred = store.get_by_prefix("glob*:").await.unwrap();
    assert_eq!(starred.len(), 1, "unexpected entries: {starred:?}");
    assert!(starred.contains_key("glob*:a"));
    assert_eq!(store.delete_by_prefix("glob*:").await.unwrap(), 1);
    assert!(store.exists("globX:a").await.unwrap(), "the decoy survives");
    assert_eq!(store.delete_by_prefix("job:").await.unwrap(), 3);
    assert!(store.get_by_prefix("job:").await.unwrap().is_empty());
    assert_eq!(store.delete_by_prefix("job:").await.unwrap(), 0);
}
#[tokio::test]
async fn store_batch_delete_counts_live_keys_once() {
    let fake = Server::start().await;
    let store = connect_store(&fake).await;
    for key in ["b:1", "b:2", "b:3"] {
        store.set(key, kv(key)).await.unwrap();
    }
    let keys: Vec<String> = ["b:1", "b:2", "b:1", "b:missing"]
        .map(str::to_owned)
        .to_vec();
    assert_eq!(store.batch_delete(&keys).await.unwrap(), 2);
    assert!(!store.exists("b:1").await.unwrap());
    assert!(!store.exists("b:2").await.unwrap());
    assert!(store.exists("b:3").await.unwrap());
    assert_eq!(store.batch_delete(&keys).await.unwrap(), 0);
}
#[tokio::test]
async fn store_pipeline_batches_in_order_and_sees_its_own_writes() {
    let fake = Server::start().await;
    let store = connect_store(&fake).await;
    store.set("before", kv("old")).await.unwrap();
    let mut pipeline = store.pipeline();
    pipeline.set("piped", kv("fresh"));
    pipeline.get("piped");
    pipeline.get("before");
    pipeline.exists("piped");
    pipeline.exists("absent");
    let expected = vec![
        None,
        Some(kv("fresh")),
        Some(kv("old")),
        Some(kv("true")),
        Some(kv("false")),
    ];
    assert_eq!(pipeline.execute().await.unwrap(), expected);
    assert!(
        pipeline.execute().await.unwrap().is_empty(),
        "the queue is cleared"
    );
    assert_eq!(store.get("piped").await.unwrap(), Some(kv("fresh")));
}
// ---------------------------------------------------------------------------
// RedisCheckpointer
// ---------------------------------------------------------------------------
#[tokio::test]
async fn checkpointer_save_and_load_preserve_every_field() {
    let fake = Server::start().await;
    let cp = connect_checkpointer(&fake).await;
    let mut metadata = BTreeMap::new();
    metadata.insert("origin".to_owned(), json!("crate-boundary"));
    let checkpoint = Checkpoint::new("team-session", CheckpointNamespace::AgentTeam, "team-1")
        .with_global_state(json!({"phase": "planning", "votes": [1, 2]}))
        .with_component_state(json!({"planner": {"done": false}}))
        .with_updates(json!({"pending": ["a", "b"]}))
        .with_metadata(metadata.clone());
    let created_at = checkpoint.created_at.clone();
    let checkpoint_id = cp.save(checkpoint).await.unwrap();
    assert!(!checkpoint_id.is_empty(), "the backend mints the id");
    let loaded = cp
        .load("team-session", &checkpoint_id)
        .await
        .unwrap()
        .expect("just saved");
    assert_eq!(loaded.session_id, "team-session");
    assert_eq!(loaded.checkpoint_id, checkpoint_id);
    assert_eq!(loaded.namespace, CheckpointNamespace::AgentTeam);
    assert_eq!(loaded.entity_id, "team-1");
    assert_eq!(loaded.created_at, created_at);
    assert_eq!(
        loaded.global_state,
        json!({"phase": "planning", "votes": [1, 2]})
    );
    assert_eq!(loaded.component_state, json!({"planner": {"done": false}}));
    assert_eq!(loaded.updates, json!({"pending": ["a", "b"]}));
    assert_eq!(loaded.metadata, metadata);
    assert_eq!(loaded.trace, None);
}
#[tokio::test]
async fn checkpointer_lists_newest_first_and_latest_picks_the_newest() {
    let fake = Server::start().await;
    let cp = connect_checkpointer(&fake).await;
    let first = save_at(cp.as_ref(), "ordered", T1, "first").await;
    let second = save_at(cp.as_ref(), "ordered", T2, "second").await;
    let third = save_at(cp.as_ref(), "ordered", T3, "third").await;
    let metas = cp.list("ordered").await.unwrap();
    let ids: Vec<&str> = metas
        .iter()
        .map(|meta| meta.checkpoint_id.as_str())
        .collect();
    assert_eq!(ids, vec![third.as_str(), second.as_str(), first.as_str()]);
    assert_eq!(metas[0].created_at, T3);
    assert_eq!(metas[0].namespace, CheckpointNamespace::Agent);
    assert_eq!(metas[0].entity_id, "agent-1");
    let latest = cp
        .latest("ordered")
        .await
        .unwrap()
        .expect("checkpoints exist");
    assert_eq!(latest.checkpoint_id, third);
    assert_eq!(latest.global_state, json!({"note": "third"}));
}
#[tokio::test]
async fn checkpointer_delete_keeps_the_session_and_release_clears_it() {
    let fake = Server::start().await;
    let cp = connect_checkpointer(&fake).await;
    let first = save_at(cp.as_ref(), "sess", T1, "first").await;
    let second = save_at(cp.as_ref(), "sess", T2, "second").await;
    assert!(cp.session_exists("sess").await.unwrap());
    cp.delete("sess", &first).await.unwrap();
    assert!(
        cp.session_exists("sess").await.unwrap(),
        "delete keeps the session"
    );
    assert!(cp.load("sess", &first).await.unwrap().is_none());
    let remaining = cp.list("sess").await.unwrap();
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].checkpoint_id, second);
    cp.release("sess").await.unwrap();
    assert!(!cp.session_exists("sess").await.unwrap());
    assert_eq!(cp.list("sess").await.unwrap(), Vec::new());
    assert!(cp.load("sess", &second).await.unwrap().is_none());
    let leftovers: Vec<String> = fake
        .state()
        .keys()
        .into_iter()
        .filter(|k| k.contains("sess"))
        .collect();
    assert!(leftovers.is_empty(), "release left {leftovers:?}");
}
#[tokio::test]
async fn checkpointer_reports_unknown_sessions_and_checkpoints() {
    let fake = Server::start().await;
    let cp = connect_checkpointer(&fake).await;
    assert!(!cp.session_exists("ghost").await.unwrap());
    assert_eq!(cp.list("ghost").await.unwrap(), Vec::new());
    assert!(cp.latest("ghost").await.unwrap().is_none());
    assert!(cp.load("ghost", "no-such-id").await.unwrap().is_none());
    // `release` of an unknown session writes nothing.
    cp.release("never-seen").await.unwrap();
    assert!(!cp.session_exists("never-seen").await.unwrap());
    assert!(fake.state().keys().is_empty(), "nothing was written");
    let saved = save_at(cp.as_ref(), "ghost", T1, "only").await;
    assert!(cp.session_exists("ghost").await.unwrap());
    assert!(cp.load("ghost", "other-id").await.unwrap().is_none());
    assert_eq!(cp.list("ghost").await.unwrap().len(), 1);
    assert_eq!(
        cp.latest("ghost").await.unwrap().unwrap().checkpoint_id,
        saved
    );
}
#[tokio::test]
async fn checkpointer_uses_the_documented_keys_and_json_payloads() {
    let fake = Server::start().await;
    let cp = connect_checkpointer(&fake).await;
    let checkpoint = Checkpoint::new("layouts", CheckpointNamespace::Workflow, "flow-9")
        .with_checkpoint_id("cp-1")
        .with_global_state(json!({"step": 3}));
    let saved = cp.save(checkpoint).await.unwrap();
    assert_eq!(saved, "cp-1", "the explicit id is honoured");
    let keys = fake.state().keys();
    let payload_key = "openjiuwen:checkpoint:layouts:cp-1".to_owned();
    let index_key = "openjiuwen:checkpoint-index:layouts".to_owned();
    assert!(keys.contains(&payload_key), "unexpected keys: {keys:?}");
    assert!(keys.contains(&index_key), "unexpected keys: {keys:?}");
    let payload = fake
        .state()
        .get(&payload_key)
        .expect("the payload key holds JSON");
    let decoded: Value = serde_json::from_slice(&payload).unwrap();
    assert_eq!(decoded["session_id"], json!("layouts"));
    assert_eq!(decoded["checkpoint_id"], json!("cp-1"));
    assert_eq!(decoded["namespace"], json!("workflow"));
    assert_eq!(decoded["entity_id"], json!("flow-9"));
    assert_eq!(decoded["global_state"], json!({"step": 3}));
    // The index is a sorted set of this session's checkpoint ids.
    let client = Client::from_url(&fake.url()).await.unwrap();
    assert_eq!(
        client.command_str(&["ZCARD", &index_key]).await.unwrap(),
        Reply::Integer(1)
    );
    let members = client
        .command_str(&["ZRANGE", &index_key, "0", "-1"])
        .await
        .unwrap();
    assert_eq!(members, Reply::Array(vec![Reply::Bulk(b"cp-1".to_vec())]));
}
// ---------------------------------------------------------------------------
// TTL
// ---------------------------------------------------------------------------
#[tokio::test]
async fn ttl_config_expires_both_the_payload_and_the_index() {
    let fake = Server::start().await;
    let config = RedisCheckpointerConfig::new(RedisConnectionConfig::new(fake.url()))
        .with_ttl(Some(RedisTTLConfig::new(Some(0.05), false)));
    let cp = RedisCheckpointer::connect(config).await.unwrap();
    let id = cp
        .save(Checkpoint::new(
            "ttl-session",
            CheckpointNamespace::Agent,
            "a-ttl",
        ))
        .await
        .unwrap();
    // `0.05` minutes rounds up to 3 s for both keys.
    let payload_key = format!("openjiuwen:checkpoint:ttl-session:{id}");
    let index_key = "openjiuwen:checkpoint-index:ttl-session".to_owned();
    for key in [payload_key, index_key] {
        let ttl = fake
            .state()
            .ttl_ms(&key)
            .unwrap_or_else(|| panic!("{key} missing"));
        assert!(
            (2_000..=3_000).contains(&ttl),
            "unexpected PTTL for {key}: {ttl}"
        );
    }
}
#[tokio::test]
async fn refresh_on_read_rearms_the_payload_and_index_ttl() {
    let fake = Server::start().await;
    let config = RedisCheckpointerConfig::new(RedisConnectionConfig::new(fake.url()))
        .with_ttl(Some(RedisTTLConfig::new(Some(0.05), true)));
    let cp = RedisCheckpointer::connect(config).await.unwrap();
    let id = cp
        .save(Checkpoint::new(
            "refresh",
            CheckpointNamespace::Agent,
            "a-refresh",
        ))
        .await
        .unwrap();
    let payload_key = format!("openjiuwen:checkpoint:refresh:{id}");
    let index_key = "openjiuwen:checkpoint-index:refresh".to_owned();
    // Shorten both leases through a second connection to the same server.
    let client = Client::from_url(&fake.url()).await.unwrap();
    client
        .command_str(&["EXPIRE", &payload_key, "1"])
        .await
        .unwrap();
    client
        .command_str(&["EXPIRE", &index_key, "1"])
        .await
        .unwrap();
    let shortened = fake
        .state()
        .ttl_ms(&payload_key)
        .expect("the payload exists");
    assert!(
        (1..=1_000).contains(&shortened),
        "unexpected PTTL {shortened}"
    );
    let loaded = cp
        .load("refresh", &id)
        .await
        .unwrap()
        .expect("a successful load");
    assert_eq!(loaded.checkpoint_id, id);
    let refreshed = fake
        .state()
        .ttl_ms(&payload_key)
        .expect("the payload exists");
    assert!(
        refreshed > 1_000,
        "load did not re-arm the payload TTL: {refreshed}"
    );
    let index = fake.state().ttl_ms(&index_key).expect("the index exists");
    assert!(index > 1_000, "load did not re-arm the index TTL: {index}");
}
// ---------------------------------------------------------------------------
// Factory and deferrals
// ---------------------------------------------------------------------------
#[tokio::test]
async fn factory_builds_a_live_redis_key_value_store() {
    let fake = Server::start().await;
    let store = build_kv_store(&StoreBackend::Redis {
        url: Some(fake.url()),
    })
    .await
    .unwrap();
    store.set("factory", kv("ok")).await.unwrap();
    assert_eq!(store.get("factory").await.unwrap(), Some(kv("ok")));
}
#[tokio::test]
async fn factory_rejects_a_missing_url_and_unavailable_redis_stores() {
    let error = expect_error(build_kv_store(&StoreBackend::Redis { url: None }).await);
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("url"), "unexpected: {error}");
    let configured = StoreBackend::Redis {
        url: Some("redis://127.0.0.1:6379/0".to_owned()),
    };
    let errors = [
        expect_error(build_document_store(&configured).await),
        expect_error(build_vector_store(&configured).await),
    ];
    for error in errors {
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        let message = error.message().to_ascii_lowercase();
        assert!(message.contains("redis"), "unexpected: {error}");
    }
}
#[tokio::test]
async fn tls_urls_are_unsupported_and_name_tls() {
    let tls_url = || "rediss://127.0.0.1:6379".to_owned();
    let store = RedisStore::connect(RedisStoreConfig {
        url: Some(tls_url()),
        ..RedisStoreConfig::default()
    });
    let config = RedisCheckpointerConfig::new(RedisConnectionConfig::new(tls_url()));
    let errors = [
        expect_error(store.await),
        expect_error(RedisCheckpointer::connect(config).await),
    ];
    for error in errors {
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        let message = error.message().to_ascii_lowercase();
        assert!(message.contains("tls"), "unexpected: {error}");
    }
}
#[tokio::test]
async fn cluster_redirections_are_unsupported_and_name_slot_routing() {
    let fake = Server::start_moved().await;
    let cluster_url = fake.url().replacen("redis://", "redis+cluster://", 1);
    let store = RedisStore::connect(RedisStoreConfig {
        url: Some(cluster_url),
        ..RedisStoreConfig::default()
    });
    let mut connection = RedisConnectionConfig::new(fake.url());
    connection.cluster_mode = Some(true);
    let checkpointer = RedisCheckpointer::connect(RedisCheckpointerConfig::new(connection));
    let errors = [expect_error(store.await), expect_error(checkpointer.await)];
    for error in errors {
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        let message = error.message().to_ascii_lowercase();
        assert!(message.contains("cluster"), "unexpected: {error}");
        assert!(
            message.contains("moved") || message.contains("ask"),
            "unexpected: {error}"
        );
    }
}
#[tokio::test]
async fn a_closed_port_is_a_transport_error() {
    let url = closed_port_url().await;
    let store = RedisStore::connect(RedisStoreConfig {
        url: Some(url.clone()),
        ..RedisStoreConfig::default()
    });
    let config = RedisCheckpointerConfig::new(RedisConnectionConfig::new(url));
    let errors = [
        expect_error(store.await),
        expect_error(RedisCheckpointer::connect(config).await),
    ];
    for error in errors {
        assert_eq!(error.kind(), ErrorKind::Transport);
    }
}
// ---------------------------------------------------------------------------
// Local fake `RESP2` server
// ---------------------------------------------------------------------------
mod fake {
    //! A compact stateful `RESP2` peer: requests are parsed with the crate's own
    //! `codec::parse_reply`, replies are hand-encoded (`encode_command` only
    //! writes requests), and expiry is lazy, so no task or clock is involved.
    //! Dropping the server signals the accept loop, which drops the listener.
    #![allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    use openjiuwen_extensions::redis::codec::{parse_reply, Reply};
    use std::collections::{BTreeMap, BTreeSet, HashMap};
    use std::net::SocketAddr;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};
    use tokio::sync::oneshot;
    use tokio::time::Instant;
    /// A running fake server; dropping it stops the loop and releases the port.
    pub(super) struct Server {
        addr: SocketAddr,
        state: State,
        stop: Option<oneshot::Sender<()>>,
    }
    impl Server {
        /// Starts a server that answers every supported command.
        pub(super) async fn start() -> Self {
            Self::start_with(false).await
        }
        /// Starts a server answering `-MOVED 3999 127.0.0.1:7000` after the handshake.
        pub(super) async fn start_moved() -> Self {
            Self::start_with(true).await
        }
        async fn start_with(moved: bool) -> Self {
            let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
            let addr = listener.local_addr().unwrap();
            let data = Arc::new(Mutex::new(Data::default()));
            let (stop, mut stopped) = oneshot::channel();
            let accept_data = Arc::clone(&data);
            tokio::spawn(async move {
                loop {
                    tokio::select! {
                        _ = &mut stopped => break,
                        accepted = listener.accept() => match accepted {
                            Ok((stream, _peer)) => {
                                let connection = Arc::clone(&accept_data);
                                tokio::spawn(serve(stream, connection, moved));
                            }
                            Err(_) => break,
                        },
                    }
                }
                // The loop owns `listener`, so leaving it releases the port.
            });
            Self {
                addr,
                state: State { data },
                stop: Some(stop),
            }
        }
        /// A `redis://` URL pointing at this server, database `0`.
        pub(super) fn url(&self) -> String {
            format!("redis://{}/0", self.addr)
        }
        /// A handle for asserting on server state.
        pub(super) fn state(&self) -> &State {
            &self.state
        }
    }
    impl Drop for Server {
        fn drop(&mut self) {
            if let Some(stop) = self.stop.take() {
                let _ = stop.send(());
            }
        }
    }
    /// Assertion handle over a running server's data.
    pub(super) struct State {
        data: Arc<Mutex<Data>>,
    }
    impl State {
        /// The stored string payload for `key`, or `None` when absent or expired.
        pub(super) fn get(&self, key: &str) -> Option<Vec<u8>> {
            self.with(|data| data.values.get(key.as_bytes()).cloned())
        }
        /// Every live key across namespaces, sorted.
        pub(super) fn keys(&self) -> Vec<String> {
            self.with(|data| {
                let keys = data.keys();
                keys.into_iter()
                    .map(|key| String::from_utf8_lossy(&key).into_owned())
                    .collect()
            })
        }
        /// Remaining TTL in milliseconds: `None` when absent, `Some(-1)` without expiry.
        pub(super) fn ttl_ms(&self, key: &str) -> Option<i64> {
            self.with(|data| {
                if !data.exists(key.as_bytes()) {
                    return None;
                }
                let remaining = data.expiries.get(key.as_bytes()).map_or(-1, |at| {
                    i64::try_from(at.saturating_duration_since(Instant::now()).as_millis())
                        .unwrap_or(i64::MAX)
                });
                Some(remaining)
            })
        }
        fn with<T>(&self, read: impl FnOnce(&mut Data) -> T) -> T {
            let mut data = self
                .data
                .lock()
                .expect("the fake state mutex is not poisoned");
            data.purge();
            read(&mut data)
        }
    }
    /// Stored data with lazy TTL expiry.
    #[derive(Default)]
    struct Data {
        values: HashMap<Vec<u8>, Vec<u8>>,
        sets: HashMap<Vec<u8>, BTreeSet<Vec<u8>>>,
        zsets: HashMap<Vec<u8>, BTreeMap<Vec<u8>, f64>>,
        expiries: HashMap<Vec<u8>, Instant>,
    }
    impl Data {
        fn purge(&mut self) {
            let now = Instant::now();
            let expired: Vec<Vec<u8>> = self
                .expiries
                .iter()
                .filter(|(_, at)| **at <= now)
                .map(|(key, _)| key.clone())
                .collect();
            for key in expired {
                self.remove(&key);
            }
        }
        fn exists(&self, key: &[u8]) -> bool {
            self.values.contains_key(key)
                || self.sets.contains_key(key)
                || self.zsets.contains_key(key)
        }
        fn remove(&mut self, key: &[u8]) -> bool {
            let value = self.values.remove(key).is_some();
            let set = self.sets.remove(key).is_some();
            let zset = self.zsets.remove(key).is_some();
            self.expiries.remove(key);
            value || set || zset
        }
        fn keys(&self) -> Vec<Vec<u8>> {
            let keys = self
                .values
                .keys()
                .chain(self.sets.keys())
                .chain(self.zsets.keys());
            let mut keys: Vec<Vec<u8>> = keys.cloned().collect();
            keys.sort();
            keys.dedup();
            keys
        }
    }
    /// Accepts one connection and answers requests until it closes.
    async fn serve(mut stream: TcpStream, state: Arc<Mutex<Data>>, moved: bool) {
        let mut buffer: Vec<u8> = Vec::new();
        let mut chunk = [0u8; 4096];
        loop {
            match parse_reply(&buffer) {
                Ok(Some((request, consumed))) => {
                    buffer.drain(..consumed);
                    let reply = request_args(&request).map_or_else(
                        || error("ERR Protocol error: not an array"),
                        |args| execute(&state, &args, moved),
                    );
                    if send_reply(&mut stream, &reply).await.is_err() {
                        return;
                    }
                }
                Ok(None) => match stream.read(&mut chunk).await {
                    Ok(read) if read > 0 => buffer.extend_from_slice(&chunk[..read]),
                    _ => return,
                },
                Err(_) => return,
            }
        }
    }
    fn request_args(request: &Reply) -> Option<Vec<Vec<u8>>> {
        let Reply::Array(items) = request else {
            return None;
        };
        items
            .iter()
            .map(|item| match item {
                Reply::Bulk(bytes) => Some(bytes.clone()),
                Reply::Simple(text) => Some(text.as_bytes().to_vec()),
                _ => None,
            })
            .collect()
    }
    // Returns the `-ERR wrong number of arguments` reply from the caller.
    macro_rules! exactly {
        ($args:expr, $count:expr, $name:expr) => {
            if $args.len() != $count {
                return wrong_arity(&$name);
            }
        };
    }
    // Returns the `-ERR wrong number of arguments` reply from the caller.
    macro_rules! at_least {
        ($args:expr, $count:expr, $name:expr) => {
            if $args.len() < $count {
                return wrong_arity(&$name);
            }
        };
    }
    /// Executes one command against the shared state.
    #[allow(clippy::too_many_lines)]
    fn execute(state: &Arc<Mutex<Data>>, args: &[Vec<u8>], moved: bool) -> Reply {
        let Some(name) = args.first().map(|arg| upper(arg)) else {
            return error("ERR Protocol error: empty command");
        };
        if moved && !matches!(name.as_str(), "AUTH" | "SELECT" | "HELLO") {
            return error("MOVED 3999 127.0.0.1:7000");
        }
        let mut guard = state.lock().expect("the fake state mutex is not poisoned");
        guard.purge();
        let data = &mut *guard;
        match name.as_str() {
            "PING" => match args.len() {
                1 => simple("PONG"),
                2 => Reply::Bulk(args[1].clone()),
                _ => wrong_arity(&name),
            },
            "ECHO" => {
                exactly!(args, 2, name);
                Reply::Bulk(args[1].clone())
            }
            "AUTH" | "SELECT" => {
                if args.len() == 2 || args.len() == 3 {
                    simple("OK")
                } else {
                    wrong_arity(&name)
                }
            }
            "GET" => {
                exactly!(args, 2, name);
                let key = args[1].as_slice();
                if data.sets.contains_key(key) || data.zsets.contains_key(key) {
                    return wrong_type();
                }
                data.values
                    .get(key)
                    .cloned()
                    .map_or(Reply::Null, Reply::Bulk)
            }
            "SET" => set(data, args),
            "EXISTS" => {
                at_least!(args, 2, name);
                let live = args[1..].iter().filter(|key| data.exists(key)).count();
                Reply::Integer(count(live))
            }
            "DEL" | "UNLINK" => {
                at_least!(args, 2, name);
                let removed = args[1..].iter().filter(|key| data.remove(key)).count();
                Reply::Integer(count(removed))
            }
            "MGET" => {
                at_least!(args, 2, name);
                let values = args[1..].iter().map(|key| {
                    data.values
                        .get(key.as_slice())
                        .cloned()
                        .map_or(Reply::Null, Reply::Bulk)
                });
                Reply::Array(values.collect())
            }
            "TTL" | "PTTL" => {
                exactly!(args, 2, name);
                ttl_reply(data, &args[1], name == "PTTL")
            }
            "EXPIRE" => {
                exactly!(args, 3, name);
                let Some(seconds) = parse_i64(&args[2]) else {
                    return error("ERR value is not an integer or out of range");
                };
                let key = args[1].as_slice();
                if !data.exists(key) {
                    return Reply::Integer(0);
                }
                if seconds <= 0 {
                    data.remove(key);
                    return Reply::Integer(1);
                }
                let lease = Duration::from_secs(u64::try_from(seconds).unwrap_or(0));
                data.expiries
                    .insert(args[1].clone(), Instant::now() + lease);
                Reply::Integer(1)
            }
            "SCAN" => scan(data, args, &name),
            "KEYS" => {
                exactly!(args, 2, name);
                let pattern = String::from_utf8_lossy(&args[1]).into_owned();
                Reply::Array(
                    matching(data, Some(&pattern))
                        .into_iter()
                        .map(Reply::Bulk)
                        .collect(),
                )
            }
            "SADD" | "SREM" | "SMEMBERS" => set_command(data, args, &name),
            "ZCARD" => {
                exactly!(args, 2, name);
                let length = data.zsets.get(args[1].as_slice()).map_or(0, BTreeMap::len);
                Reply::Integer(count(length))
            }
            "ZADD" | "ZRANGE" | "ZREM" => zset_command(data, args, &name),
            "FLUSHDB" => {
                data.values.clear();
                data.sets.clear();
                data.zsets.clear();
                data.expiries.clear();
                simple("OK")
            }
            "TYPE" => {
                exactly!(args, 2, name);
                let key = args[1].as_slice();
                let kind = if data.values.contains_key(key) {
                    "string"
                } else if data.sets.contains_key(key) {
                    "set"
                } else if data.zsets.contains_key(key) {
                    "zset"
                } else {
                    "none"
                };
                simple(kind)
            }
            _ => error(&format!("ERR unknown command '{name}'")),
        }
    }
    /// Handles `SET key value [NX] [KEEPTTL] [EX s | PX ms]`; a plain `SET` drops
    /// any previous TTL, which the kernel `KvStore` contract depends on.
    fn set(data: &mut Data, args: &[Vec<u8>]) -> Reply {
        if args.len() < 3 {
            return wrong_arity("SET");
        }
        let key = args[1].clone();
        let (mut nx, mut keep_ttl, mut expiry) = (false, false, None);
        let mut index = 3;
        while index < args.len() {
            let option = upper(&args[index]);
            match option.as_str() {
                "NX" => nx = true,
                "KEEPTTL" => keep_ttl = true,
                "EX" | "PX" => {
                    let Some(amount) = args.get(index + 1).and_then(|raw| parse_i64(raw)) else {
                        return error("ERR value is not an integer or out of range");
                    };
                    let amount = u64::try_from(amount).unwrap_or(0);
                    let lease = if option == "PX" {
                        Duration::from_millis(amount)
                    } else {
                        Duration::from_secs(amount)
                    };
                    expiry = Some(Instant::now() + lease);
                    index += 1;
                }
                _ => return error("ERR syntax error"),
            }
            index += 1;
        }
        if nx && data.exists(&key) {
            return Reply::Null;
        }
        data.sets.remove(&key);
        data.zsets.remove(&key);
        data.values.insert(key.clone(), args[2].clone());
        if let Some(at) = expiry {
            data.expiries.insert(key, at);
        } else if !keep_ttl {
            data.expiries.remove(&key);
        }
        simple("OK")
    }
    /// Handles `SCAN cursor [MATCH glob] [COUNT n]`: one batch, cursor `0`.
    fn scan(data: &Data, args: &[Vec<u8>], name: &str) -> Reply {
        at_least!(args, 2, name);
        let mut pattern: Option<String> = None;
        let mut index = 2;
        while index < args.len() {
            let option = upper(&args[index]);
            match option.as_str() {
                "MATCH" | "COUNT" => {
                    let Some(raw) = args.get(index + 1) else {
                        return error("ERR syntax error");
                    };
                    if option == "MATCH" {
                        pattern = Some(String::from_utf8_lossy(raw).into_owned());
                    }
                    index += 1;
                }
                _ => return error("ERR syntax error"),
            }
            index += 1;
        }
        let keys = matching(data, pattern.as_deref())
            .into_iter()
            .map(Reply::Bulk)
            .collect();
        Reply::Array(vec![Reply::Bulk(b"0".to_vec()), Reply::Array(keys)])
    }
    /// Handles `SADD`/`SREM`/`SMEMBERS`.
    fn set_command(data: &mut Data, args: &[Vec<u8>], name: &str) -> Reply {
        if name == "SMEMBERS" {
            exactly!(args, 2, name);
            let members = data
                .sets
                .get(args[1].as_slice())
                .map_or_else(Vec::new, |set| {
                    set.iter().cloned().map(Reply::Bulk).collect()
                });
            return Reply::Array(members);
        }
        at_least!(args, 3, name);
        let key = args[1].clone();
        if name == "SADD" {
            let set = data.sets.entry(key).or_default();
            let added = args[2..]
                .iter()
                .filter(|member| set.insert((*member).clone()))
                .count();
            return Reply::Integer(count(added));
        }
        let removed = data.sets.get_mut(&key).map_or(0, |set| {
            args[2..]
                .iter()
                .filter(|member| set.remove(member.as_slice()))
                .count()
        });
        if data.sets.get(&key).is_some_and(BTreeSet::is_empty) {
            data.sets.remove(&key);
        }
        Reply::Integer(count(removed))
    }
    /// Handles `ZADD`/`ZRANGE`/`ZREM`; ranges are ordered by score then member.
    fn zset_command(data: &mut Data, args: &[Vec<u8>], name: &str) -> Reply {
        let key = args.get(1).cloned().unwrap_or_default();
        if name == "ZADD" {
            if args.len() < 4 || (args.len() - 2) % 2 != 0 {
                return wrong_arity(name);
            }
            let mut added = 0;
            for pair in args[2..].chunks(2) {
                let Some(score) = parse_f64(&pair[0]) else {
                    return error("ERR value is not a valid float");
                };
                if data
                    .zsets
                    .entry(key.clone())
                    .or_default()
                    .insert(pair[1].clone(), score)
                    .is_none()
                {
                    added += 1;
                }
            }
            return Reply::Integer(count(added));
        }
        if name == "ZREM" {
            at_least!(args, 3, name);
            let removed = data.zsets.get_mut(&key).map_or(0, |zset| {
                args[2..]
                    .iter()
                    .filter(|member| zset.remove(member.as_slice()).is_some())
                    .count()
            });
            if data.zsets.get(&key).is_some_and(BTreeMap::is_empty) {
                data.zsets.remove(&key);
            }
            return Reply::Integer(count(removed));
        }
        exactly!(args, 4, name);
        let (Some(start), Some(stop)) = (
            args.get(2).and_then(|raw| parse_i64(raw)),
            args.get(3).and_then(|raw| parse_i64(raw)),
        ) else {
            return error("ERR value is not an integer or out of range");
        };
        let mut entries: Vec<(&Vec<u8>, f64)> =
            data.zsets
                .get(key.as_slice())
                .map_or_else(Vec::new, |zset| {
                    zset.iter()
                        .map(|(member, score)| (member, *score))
                        .collect()
                });
        entries.sort_by(|left, right| {
            left.1
                .partial_cmp(&right.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.0.cmp(right.0))
        });
        let length = count(entries.len());
        let resolve = |index: i64| if index < 0 { length + index } else { index };
        let (start, stop) = (resolve(start).max(0), resolve(stop).min(length - 1));
        if stop < start || start >= length {
            return Reply::Array(Vec::new());
        }
        let (skip, take) = (
            usize::try_from(start).unwrap_or(0),
            usize::try_from(stop - start).unwrap_or(0),
        );
        let members = entries
            .iter()
            .skip(skip)
            .take(take + 1)
            .map(|(member, _)| Reply::Bulk((*member).clone()));
        Reply::Array(members.collect())
    }
    /// Upper-cases a byte argument for command/option matching.
    fn upper(value: &[u8]) -> String {
        String::from_utf8_lossy(value).to_ascii_uppercase()
    }
    /// Parses a byte argument as an `i64`.
    fn parse_i64(value: &[u8]) -> Option<i64> {
        std::str::from_utf8(value).ok()?.parse().ok()
    }
    /// Parses a byte argument as a finite `f64`.
    fn parse_f64(value: &[u8]) -> Option<f64> {
        let parsed = std::str::from_utf8(value).ok()?.parse::<f64>().ok()?;
        parsed.is_finite().then_some(parsed)
    }
    /// Converts a count to `i64` without panicking.
    fn count(count: usize) -> i64 {
        i64::try_from(count).unwrap_or(i64::MAX)
    }
    fn simple(text: &str) -> Reply {
        Reply::Simple(text.to_owned())
    }
    fn error(message: &str) -> Reply {
        Reply::Error(message.to_owned())
    }
    fn wrong_arity(name: &str) -> Reply {
        error(&format!(
            "ERR wrong number of arguments for '{name}' command"
        ))
    }
    fn wrong_type() -> Reply {
        error("WRONGTYPE Operation against a key holding the wrong kind of value")
    }
    /// The `-2`/`-1`/remaining TTL reply.
    fn ttl_reply(data: &Data, key: &[u8], milliseconds: bool) -> Reply {
        if !data.exists(key) {
            return Reply::Integer(-2);
        }
        let Some(at) = data.expiries.get(key) else {
            return Reply::Integer(-1);
        };
        let remaining = at.saturating_duration_since(Instant::now());
        let value = if milliseconds {
            remaining.as_millis()
        } else {
            u128::from(remaining.as_secs())
        };
        Reply::Integer(i64::try_from(value).unwrap_or(i64::MAX))
    }
    /// The live keys matching an optional glob pattern.
    fn matching(data: &Data, pattern: Option<&str>) -> Vec<Vec<u8>> {
        data.keys()
            .into_iter()
            .filter(|key| pattern.is_none_or(|glob| glob_match(glob.as_bytes(), key)))
            .collect()
    }
    /// A glob matcher for `*`, `?`, and backslash escapes, so the store's escaped
    /// prefix still reaches a literal match.
    fn glob_match(pattern: &[u8], text: &[u8]) -> bool {
        match pattern.first() {
            None => text.is_empty(),
            Some(b'\\') => pattern.get(1).is_some_and(|expected| {
                text.first() == Some(expected) && glob_match(&pattern[2..], &text[1..])
            }),
            Some(b'*') => {
                let rest = &pattern[1..];
                let mut index = 0;
                loop {
                    if glob_match(rest, &text[index..]) {
                        return true;
                    }
                    if index == text.len() {
                        return false;
                    }
                    index += 1;
                }
            }
            Some(b'?') => !text.is_empty() && glob_match(&pattern[1..], &text[1..]),
            Some(expected) => {
                text.first() == Some(expected) && glob_match(&pattern[1..], &text[1..])
            }
        }
    }
    /// Encodes a reply as `RESP2`.
    fn encode(reply: &Reply) -> Vec<u8> {
        match reply {
            Reply::Simple(text) => line(b'+', text.as_bytes()),
            Reply::Error(message) => line(b'-', message.as_bytes()),
            Reply::Integer(value) => format!(":{value}\r\n").into_bytes(),
            Reply::Bulk(bytes) => {
                let mut out = format!("${}\r\n", bytes.len()).into_bytes();
                out.extend_from_slice(bytes);
                out.extend_from_slice(b"\r\n");
                out
            }
            Reply::Null => b"$-1\r\n".to_vec(),
            Reply::Array(items) => {
                let mut out = format!("*{}\r\n", items.len()).into_bytes();
                for item in items {
                    out.extend_from_slice(&encode(item));
                }
                out
            }
            other => line(
                b'-',
                format!("ERR unsupported reply {}", other.kind_name()).as_bytes(),
            ),
        }
    }
    fn line(tag: u8, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(payload.len() + 3);
        out.push(tag);
        out.extend_from_slice(payload);
        out.extend_from_slice(b"\r\n");
        out
    }
    /// Writes one encoded reply to the connection.
    async fn send_reply(stream: &mut TcpStream, reply: &Reply) -> std::io::Result<()> {
        stream.write_all(&encode(reply)).await?;
        stream.flush().await
    }
}
