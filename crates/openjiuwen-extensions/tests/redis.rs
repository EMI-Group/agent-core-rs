//! Crate-boundary integration suite for the Redis adapters.
//!
//! Every assertion enters through a public path (`openjiuwen_extensions::redis`,
//! `store::redis`, `store::build_kv_store`, `checkpointer::redis`), so a broken
//! re-export, a changed factory signature, or a lost deferral contract fails here
//! before a consumer notices. The peer these tests talk to is the [`fake`] server
//! in `tests/redis/fake.rs`: it binds `127.0.0.1:0`, keeps its data in memory, and
//! answers `RESP2`, so the suite is local, offline, and deterministic.
//!
//! One section per surface: codec, store, checkpointer, TTL, factory/deferrals.
//! Without the `redis` feature this file compiles to an empty test binary.
#![cfg(feature = "redis")]
#![allow(clippy::unwrap_used, clippy::expect_used)]
#[path = "redis/fake.rs"]
mod fake;
use fake::Server;
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_core::foundation::store::{
    CollectionSchema, FieldSchema, Metric, SortOrder, VectorDataType,
};
use openjiuwen_extensions::checkpointer::redis::{
    RedisCheckpointer, RedisCheckpointerConfig, RedisConnectionConfig, RedisTTLConfig,
};
use openjiuwen_extensions::checkpointer::{Checkpoint, CheckpointNamespace, Checkpointer};
use openjiuwen_extensions::redis::codec::{encode_command, parse_reply, Reply};
use openjiuwen_extensions::redis::Client;
use openjiuwen_extensions::store::redis::{
    RedisDocumentStore, RedisStore, RedisStoreConfig, RedisVectorStore,
};
use openjiuwen_extensions::store::{
    build_document_store, build_kv_store, build_vector_store, Document, DocumentStore,
    InMemoryDocumentStore, KvStore, KvValue, Query, StoreBackend, VectorQuery, VectorStore,
};
use serde_json::{json, Map, Value};
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
async fn factory_rejects_a_missing_url_for_every_redis_store() {
    let backend = StoreBackend::Redis { url: None };
    let errors = [
        expect_error(build_kv_store(&backend).await),
        expect_error(build_document_store(&backend).await),
        expect_error(build_vector_store(&backend).await),
    ];
    for error in errors {
        assert_eq!(error.kind(), ErrorKind::Validation);
        let message = error.message().to_ascii_lowercase();
        assert!(message.contains("url"), "unexpected: {error}");
        assert!(message.contains("redis"), "unexpected: {error}");
    }
}
#[tokio::test]
async fn factory_connects_the_document_and_vector_stores_eagerly() {
    // The three stores share the `connect` contract: an unreachable peer fails
    // with a transport error before the caller stores anything.
    let url = closed_port_url().await;
    let backend = StoreBackend::Redis { url: Some(url) };
    let errors = [
        expect_error(build_kv_store(&backend).await),
        expect_error(build_document_store(&backend).await),
        expect_error(build_vector_store(&backend).await),
    ];
    for error in errors {
        assert_eq!(error.kind(), ErrorKind::Transport, "unexpected: {error}");
    }
}
#[tokio::test]
async fn factory_builds_live_redis_document_and_vector_stores() {
    let fake = Server::start().await;
    let backend = StoreBackend::Redis {
        url: Some(fake.url()),
    };
    let documents = build_document_store(&backend).await.unwrap();
    documents.create_collection("docs").await.unwrap();
    assert_eq!(
        documents.list_collections().await.unwrap(),
        vec!["docs".to_owned()]
    );
    let vectors = build_vector_store(&backend).await.unwrap();
    let schema = CollectionSchema::new(vec![
        FieldSchema {
            is_primary: true,
            ..FieldSchema::new("id", VectorDataType::Varchar)
        },
        FieldSchema {
            dim: Some(2),
            ..FieldSchema::new("embedding", VectorDataType::FloatVector)
        },
    ]);
    vectors
        .create_collection("embeddings", schema)
        .await
        .unwrap();
    assert!(vectors.collection_exists("embeddings").await.unwrap());
}
#[tokio::test]
async fn tls_urls_perform_a_handshake_and_name_tls() {
    // `rediss://` really uses TLS: against a plain RESP peer the handshake fails as a
    // transport error naming TLS, never as an unsupported feature.
    let fake = Server::start().await;
    let tls_url = format!("rediss://{}/0", fake.addr());
    let store = RedisStore::connect(RedisStoreConfig {
        url: Some(tls_url.clone()),
        ..RedisStoreConfig::default()
    });
    let config = RedisCheckpointerConfig::new(RedisConnectionConfig::new(&tls_url));
    let errors = [
        expect_error(store.await),
        expect_error(RedisCheckpointer::connect(config).await),
    ];
    for error in errors {
        assert_eq!(error.kind(), ErrorKind::Transport);
        let message = error.message().to_ascii_lowercase();
        assert!(message.contains("tls"), "unexpected: {error}");
    }
}
#[tokio::test]
async fn cluster_mode_is_verified_against_the_seed_node() {
    // A standalone node is no longer accepted in cluster mode.
    let fake = Server::start().await;
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
        assert_eq!(error.kind(), ErrorKind::Validation);
        let message = error.message();
        assert!(message.contains("cluster_enabled:1"), "unexpected: {error}");
    }
}

#[tokio::test]
async fn cluster_redirections_are_followed_to_the_reported_node() {
    // The seed reports `MOVED` for every command; the client must re-issue the command
    // on the reported node so the write lands there.
    let owner = Server::start().await;
    let seed = Server::start_moved_to(owner.addr()).await;
    let client = Client::from_url(&seed.url()).await.unwrap();
    let written = client.command_str(&["SET", "redirected", "value"]).await;
    assert_eq!(written.unwrap(), Reply::Simple("OK".to_owned()));
    assert_eq!(
        owner.state().get("redirected"),
        Some(b"value".to_vec()),
        "the command must run on the redirected node"
    );
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
// RedisDocumentStore (kernel `DocumentStore` contract)
// ---------------------------------------------------------------------------
/// Connects a document store through the crate boundary.
async fn connect_documents(fake: &Server) -> Arc<dyn DocumentStore> {
    let config = RedisStoreConfig {
        url: Some(fake.url()),
        ..RedisStoreConfig::default()
    };
    RedisDocumentStore::connect(config).await.unwrap()
}
/// Connects a vector store through the crate boundary.
async fn connect_vectors(fake: &Server) -> Arc<dyn VectorStore> {
    let config = RedisStoreConfig {
        url: Some(fake.url()),
        ..RedisStoreConfig::default()
    };
    RedisVectorStore::connect(config).await.unwrap()
}
/// A configuration carrying only `url`.
fn url_config(url: &str) -> RedisStoreConfig {
    RedisStoreConfig {
        url: Some(url.to_owned()),
        ..RedisStoreConfig::default()
    }
}
/// A document with a name and an age, the fixture the query tests use.
fn person(id: &str, name: &str, age: i64) -> Document {
    Document::new(id)
        .with_field("name", name)
        .with_field("age", age)
}
/// The ids of `documents`, in order.
fn doc_ids(documents: &[Document]) -> Vec<String> {
    documents.iter().map(|doc| doc.id.clone()).collect()
}
/// The schema every vector suite uses: keyword primary key, one 3-dimensional
/// embedding, and a `kind` field that carries a `default_value`.
fn vector_schema() -> CollectionSchema {
    CollectionSchema::new(vec![
        FieldSchema {
            is_primary: true,
            ..FieldSchema::new("id", VectorDataType::Varchar)
        },
        FieldSchema {
            dim: Some(3),
            ..FieldSchema::new("embedding", VectorDataType::FloatVector)
        },
        FieldSchema {
            default_value: Some(json!("note")),
            ..FieldSchema::new("kind", VectorDataType::Varchar)
        },
    ])
}
/// The JSON object of `value`, for the document fixtures.
fn obj(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(map) => map,
        _ => Map::new(),
    }
}
/// A vector document with a three-component embedding.
fn vector_doc(id: &str, embedding: [f32; 3]) -> Map<String, Value> {
    obj(json!({
        "id": id,
        "embedding": embedding.to_vec(),
    }))
}
/// The primary keys of `hits`, in order.
fn hit_ids(hits: &[openjiuwen_extensions::store::VectorSearchResult]) -> Vec<String> {
    hits.iter()
        .map(|hit| {
            hit.fields
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned()
        })
        .collect()
}
#[tokio::test]
async fn document_store_round_trips_collections_and_documents() {
    let fake = Server::start().await;
    let store = connect_documents(&fake).await;
    assert_eq!(
        store.list_collections().await.unwrap(),
        Vec::<String>::new(),
        "a fresh peer holds no collections"
    );

    // Creating is idempotent and listed in ascending order; `upsert` registers.
    store.create_collection("beta").await.unwrap();
    store.create_collection("alpha").await.unwrap();
    store.create_collection("alpha").await.unwrap();
    assert_eq!(store.list_collections().await.unwrap(), ["alpha", "beta"]);
    assert_eq!(
        store
            .upsert("gamma", vec![person("g", "Gina", 30)])
            .await
            .unwrap(),
        1
    );
    assert_eq!(
        store.list_collections().await.unwrap(),
        ["alpha", "beta", "gamma"]
    );

    // Every JSON shape survives the round trip, nested members included.
    let nested = Document::new("nested")
        .with_field("meta", json!({"owner": "ada", "tags": ["rust", "redis"]}))
        .with_field("ratio", 0.5)
        .with_field("count", 9_007_199_254_740_991_i64)
        .with_field("flag", true);
    store
        .upsert("docs", vec![nested.clone(), person("a", "Ada", 36)])
        .await
        .unwrap();
    assert_eq!(store.get("docs", "nested").await.unwrap(), Some(nested));
    let ada = store.get("docs", "a").await.unwrap().expect("stored");
    assert_eq!(ada.field("name"), Some(&json!("Ada")));
    assert_eq!(ada.field("age"), Some(&json!(36)));

    // A second upsert replaces the document rather than merging into it.
    store
        .upsert("docs", vec![Document::new("a").with_field("age", 37)])
        .await
        .unwrap();
    let replaced = store.get("docs", "a").await.unwrap().expect("replaced");
    assert_eq!(replaced.field("age"), Some(&json!(37)));
    assert_eq!(replaced.field("name"), None);

    // Reads, filters, ordering, and counts all come from the kernel reference.
    let thirty_plus = Query {
        filter: json!({"age": {"$gte": 30}}),
        ..Query::default()
    };
    assert_eq!(
        doc_ids(&store.query("docs", &thirty_plus).await.unwrap()),
        ["a"]
    );
    assert_eq!(store.count("docs", &thirty_plus).await.unwrap(), 1);
    let ordered = Query {
        order_by: vec![("id".to_string(), SortOrder::Desc)],
        ..Query::default()
    };
    assert_eq!(
        doc_ids(&store.query("docs", &ordered).await.unwrap()),
        ["nested", "a"]
    );

    // `delete` honors the filter only, and repeats remove nothing.
    assert_eq!(store.delete("docs", &thirty_plus).await.unwrap(), 1);
    assert_eq!(store.delete("docs", &thirty_plus).await.unwrap(), 0);
    assert!(store.get("docs", "a").await.unwrap().is_none());
    assert_eq!(store.count("docs", &Query::default()).await.unwrap(), 1);

    // Dropping is idempotent and takes the document keys with it.
    store.drop_collection("gamma").await.unwrap();
    store.drop_collection("gamma").await.unwrap();
    assert!(!store
        .list_collections()
        .await
        .unwrap()
        .contains(&"gamma".to_owned()));
    assert!(store.get("gamma", "g").await.unwrap().is_none());
    assert!(
        fake.state().keys().iter().all(|key| !key.contains("gamma")),
        "the dropped collection keeps keys: {:?}",
        fake.state().keys()
    );
}
#[tokio::test]
async fn document_store_query_windows_match_the_reference_store() {
    let fake = Server::start().await;
    let store = connect_documents(&fake).await;
    let reference = InMemoryDocumentStore::new();
    let documents: Vec<Document> = (0..25)
        .map(|index| {
            person(
                &format!("doc-{index:02}"),
                &format!("name-{index:02}"),
                i64::from(index % 5) * 10,
            )
        })
        .collect();
    store.upsert("paged", documents.clone()).await.unwrap();
    reference.upsert("paged", documents).await.unwrap();

    // Paging across more than one page, a filtered multi-key order, and an
    // unpaged scan must all return the identical window the reference returns.
    let queries = [
        Query {
            order_by: vec![("id".to_string(), SortOrder::Asc)],
            offset: 5,
            limit: Some(10),
            ..Query::default()
        },
        Query {
            order_by: vec![("id".to_string(), SortOrder::Asc)],
            offset: 20,
            limit: Some(10),
            ..Query::default()
        },
        Query {
            filter: json!({"age": {"$gte": 20}}),
            order_by: vec![
                ("age".to_string(), SortOrder::Desc),
                ("id".to_string(), SortOrder::Asc),
            ],
            offset: 2,
            limit: Some(7),
        },
        Query {
            order_by: vec![("age".to_string(), SortOrder::Asc)],
            ..Query::default()
        },
    ];
    for query in &queries {
        let expected = reference.query("paged", query).await.unwrap();
        let actual = store.query("paged", query).await.unwrap();
        assert_eq!(doc_ids(&actual), doc_ids(&expected), "query {query:?}");
    }
    assert_eq!(store.count("paged", &Query::default()).await.unwrap(), 25);
}
#[tokio::test]
async fn document_store_rejects_invalid_names_and_unreachable_peers() {
    // Construction validates the URL but performs no I/O.
    let invalid_scheme = expect_error(RedisDocumentStore::new(url_config("https://127.0.0.1:1")));
    assert_eq!(invalid_scheme.kind(), ErrorKind::Validation);
    assert!(
        invalid_scheme.message().contains("redis://"),
        "unexpected: {invalid_scheme}"
    );
    let missing = expect_error(RedisDocumentStore::new(RedisStoreConfig::default()));
    assert_eq!(missing.kind(), ErrorKind::Validation);
    let fake = Server::start().await;
    // `rediss://` performs a real TLS handshake when connecting, so a plain RESP peer
    // fails it with a transport error naming TLS (construction performs no I/O).
    let tls = RedisDocumentStore::connect(url_config(&format!("rediss://{}/0", fake.addr())))
        .await
        .err()
        .expect("a plain listener cannot complete a TLS handshake");
    assert_eq!(tls.kind(), ErrorKind::Transport);
    assert!(
        tls.message().to_ascii_lowercase().contains("tls"),
        "unexpected: {tls}"
    );
    let store = connect_documents(&fake).await;
    // The kernel's collection-name rule is replicated once and shared, so both
    // stores reject the same names with a validation error naming the collection.
    for name in ["", ".", "..", "a/b", "a\\b", "nul\0byte"] {
        let error = store
            .create_collection(name)
            .await
            .expect_err("invalid collection name");
        assert_eq!(error.kind(), ErrorKind::Validation, "name {name:?}");
        assert!(
            error.message().contains("invalid collection name"),
            "unexpected: {error}"
        );
    }
    let error = store
        .get("a/b", "x")
        .await
        .expect_err("invalid collection name");
    assert_eq!(error.kind(), ErrorKind::Validation);

    // A refused connection surfaces as a transport error, never a panic.
    let error = RedisDocumentStore::connect(url_config(&closed_port_url().await))
        .await
        .err()
        .expect("an unreachable peer must fail");
    assert_eq!(error.kind(), ErrorKind::Transport);
}
// ---------------------------------------------------------------------------
// RedisVectorStore (kernel `VectorStore` contract)
// ---------------------------------------------------------------------------
#[tokio::test]
async fn vector_store_round_trips_schema_documents_and_searches() {
    let fake = Server::start().await;
    let store = connect_vectors(&fake).await;
    assert!(!store.collection_exists("docs").await.unwrap());
    assert_eq!(store.get_schema("docs").await.unwrap(), None);

    store
        .create_collection("docs", vector_schema())
        .await
        .unwrap();
    // Creating an existing collection is a validation error, like the reference.
    let duplicate = store
        .create_collection("docs", vector_schema())
        .await
        .expect_err("duplicate collection");
    assert_eq!(duplicate.kind(), ErrorKind::Validation);
    assert!(
        duplicate.message().contains("docs"),
        "unexpected: {duplicate}"
    );
    assert!(store.collection_exists("docs").await.unwrap());
    // The schema round-trips exactly, flags and dimensions included.
    let schema = store.get_schema("docs").await.unwrap().expect("schema");
    assert_eq!(schema, vector_schema());
    assert_eq!(schema.fields[1].dim, Some(3));
    assert_eq!(
        schema.fields[2].default_value,
        Some(json!("note")),
        "the field default survives the round trip"
    );
    assert_eq!(store.list_collection_names().await.unwrap(), ["docs"]);

    // `add_docs` fills the default and stores the documents.
    assert_eq!(
        store
            .add_docs(
                "docs",
                vec![
                    vector_doc("a", [1.0, 0.0, 0.0]),
                    vector_doc("b", [0.0, 1.0, 0.0]),
                    vector_doc("c", [-1.0, 0.0, 0.0]),
                ],
            )
            .await
            .unwrap(),
        3
    );
    let hits = store
        .search("docs", &VectorQuery::new(vec![1.0, 0.0, 0.0]).with_top_k(2))
        .await
        .unwrap();
    assert_eq!(hit_ids(&hits).first().map(String::as_str), Some("a"));
    assert_eq!(hits.len(), 2, "top_k truncates");
    assert_eq!(hits[0].fields["kind"], "note", "the default was applied");

    // Filters and paging arrive through the kernel reference untouched.
    let filtered = store
        .search(
            "docs",
            &VectorQuery::new(vec![0.0, 1.0, 0.0])
                .with_top_k(5)
                .with_filters(json!({"id": {"$in": ["b", "c"]}})),
        )
        .await
        .unwrap();
    assert_eq!(hit_ids(&filtered), ["b", "c"]);

    // Deletions by id and by filter both report the reference count.
    assert_eq!(
        store
            .delete_docs_by_ids("docs", &["a".to_string(), "missing".to_string()])
            .await
            .unwrap(),
        1
    );
    assert_eq!(
        store
            .delete_docs_by_filters("docs", &json!({"kind": "note"}))
            .await
            .unwrap(),
        2
    );
    assert_eq!(
        store
            .search("docs", &VectorQuery::new(vec![1.0, 0.0, 0.0]))
            .await
            .unwrap()
            .len(),
        0
    );

    // Deleting a collection removes the documents and the schema.
    store.delete_collection("docs").await.unwrap();
    store.delete_collection("docs").await.unwrap();
    assert!(!store.collection_exists("docs").await.unwrap());
    assert_eq!(store.get_schema("docs").await.unwrap(), None);
    assert_eq!(
        store.list_collection_names().await.unwrap(),
        Vec::<String>::new()
    );
    let error = store
        .search("docs", &VectorQuery::new(vec![1.0, 0.0, 0.0]))
        .await
        .expect_err("missing collection");
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(error.message().contains("docs"), "unexpected: {error}");
}
#[tokio::test]
async fn vector_store_metric_defaults_to_cosine_and_can_be_changed() {
    let fake = Server::start().await;
    // The default metric is cosine; `with_metric` swaps it at the store level
    // because the kernel schema carries no metric field.
    let cosine = RedisVectorStore::new(url_config(&fake.url())).unwrap();
    assert_eq!(cosine.metric(), Metric::Cosine);
    let euclidean = RedisVectorStore::new(url_config(&fake.url()))
        .unwrap()
        .with_metric(Metric::Euclidean);
    assert_eq!(euclidean.metric(), Metric::Euclidean);
    cosine
        .create_collection("docs", vector_schema())
        .await
        .unwrap();
    cosine
        .add_docs(
            "docs",
            vec![
                vector_doc("near", [1.0, 0.0, 0.0]),
                vector_doc("far", [0.0, 0.0, 1.0]),
            ],
        )
        .await
        .unwrap();

    let query = VectorQuery::new(vec![1.0, 0.0, 0.0]).with_top_k(2);
    let by_cosine = cosine.search("docs", &query).await.unwrap();
    let by_euclidean = euclidean.search("docs", &query).await.unwrap();
    assert_eq!(hit_ids(&by_cosine), hit_ids(&by_euclidean), "same ranking");
    let expected = Metric::Cosine.score(&[1.0, 0.0, 0.0], &[0.0, 0.0, 1.0]);
    assert!(
        (by_cosine[1].score - expected).abs() < 1e-6,
        "cosine scores disagree with `Metric::score`: {by_cosine:?}"
    );
    let expected = Metric::Euclidean.score(&[1.0, 0.0, 0.0], &[0.0, 0.0, 1.0]);
    assert!(
        (by_euclidean[1].score - expected).abs() < 1e-6,
        "euclidean scores disagree with `Metric::score`: {by_euclidean:?}"
    );
    assert_ne!(
        by_cosine[1].score, by_euclidean[1].score,
        "the metric must change the score"
    );
    // `connect` hands out the default-metric store as a trait object.
    let connected = connect_vectors(&fake).await;
    let via_trait = connected.search("docs", &query).await.unwrap();
    assert_eq!(hit_ids(&via_trait), hit_ids(&by_cosine));
}
#[tokio::test]
async fn vector_store_rejects_invalid_collections_and_unreachable_peers() {
    let fake = Server::start().await;
    let store = connect_vectors(&fake).await;

    // An invalid schema is rejected before anything is written.
    let invalid = CollectionSchema::new(vec![
        FieldSchema {
            is_primary: true,
            ..FieldSchema::new("id", VectorDataType::Varchar)
        },
        FieldSchema::new("embedding", VectorDataType::FloatVector),
    ]);
    let error = store
        .create_collection("broken", invalid)
        .await
        .expect_err("a vector field without `dim` is invalid");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(!store.collection_exists("broken").await.unwrap());

    // The shared collection-name rule is applied here too.
    let error = store
        .create_collection("a/b", vector_schema())
        .await
        .expect_err("invalid collection name");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.message().contains("invalid collection name"),
        "unexpected: {error}"
    );

    // Validation failures from the reference store surface unchanged and persist
    // nothing: a duplicate primary key aborts the whole batch.
    store
        .create_collection("docs", vector_schema())
        .await
        .unwrap();
    let error = store
        .add_docs(
            "docs",
            vec![
                vector_doc("a", [1.0, 0.0, 0.0]),
                vector_doc("a", [0.0, 1.0, 0.0]),
            ],
        )
        .await
        .expect_err("duplicate primary key");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("duplicate"), "unexpected: {error}");
    assert_eq!(
        store
            .search("docs", &VectorQuery::new(vec![1.0, 0.0, 0.0]))
            .await
            .unwrap()
            .len(),
        0,
        "a rejected batch persists nothing"
    );

    // Construction validates without I/O; an unreachable peer is a transport
    // error when connecting.
    let invalid_scheme = expect_error(RedisVectorStore::new(url_config("https://127.0.0.1:1")));
    assert_eq!(invalid_scheme.kind(), ErrorKind::Validation);
    let error = RedisVectorStore::connect(url_config(&closed_port_url().await))
        .await
        .err()
        .expect("an unreachable peer must fail");
    assert_eq!(error.kind(), ErrorKind::Transport);
}
