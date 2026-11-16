// ports tests/unit_tests/core/foundation/store/test_in_memory_kv_store.py
// (adapted to the Rust `KvStore` contract shared by every backend)
#![allow(clippy::unwrap_used, clippy::expect_used)]

//! The [`crate::store::KvStore`] contract against every key-value backend.
//!
//! Covers text and byte values, missing keys, overwrites, TTLs and
//! `exclusive_set`, prefix scans, multi-key reads and deletes, and pipeline
//! ordering. File persistence is verified separately because only the
//! local-filesystem backend can be reopened.

use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tempfile::TempDir;

use crate::store::{InMemoryKvStore, KvStore, KvValue, LocalFileKvStore};

/// A payload that is not valid UTF-8, to prove byte values stay byte values.
const BINARY: [u8; 4] = [0x00, 0xff, 0xfe, 0x80];

/// How long the contract lets a TTL entry stay live before it is expected to be gone.
const TTL: Duration = Duration::from_millis(40);

/// How long the contract waits for that entry to expire (a 20ms margin).
const PAST_TTL: Duration = Duration::from_millis(60);

/// A lease deadline so far away that "still live" assertions cannot race it.
///
/// Several contract steps write a lease and then assert that it is visible and that
/// writers are refused. A sub-second TTL makes that assertion depend on how quickly
/// the test thread is scheduled, which flaked under parallel load; a one-minute
/// deadline does not.
const LIVE_TTL: Duration = Duration::from_secs(60);

#[tokio::test]
async fn in_memory_satisfies_the_kv_contract() {
    assert_kv_contract(&InMemoryKvStore::new()).await;
}

#[tokio::test]
async fn local_file_satisfies_the_kv_contract() {
    let directory = TempDir::new().unwrap();
    let store = LocalFileKvStore::new(directory.path()).await.unwrap();
    assert_kv_contract(&store).await;
}

#[tokio::test]
async fn local_file_persists_values_and_ttls_across_a_reopen() {
    let directory = TempDir::new().unwrap();
    let root = directory.path().join("kv");
    {
        let store = LocalFileKvStore::new(&root).await.unwrap();
        store.set("text", KvValue::text("hello")).await.unwrap();
        store
            .set("bytes", KvValue::bytes(BINARY.to_vec()))
            .await
            .unwrap();
        // A deadline a minute away keeps the lease live across the reopen below; a
        // sub-second window could elapse while the store is reopened under load.
        assert!(store
            .exclusive_set("lock", KvValue::text("held"), Some(LIVE_TTL))
            .await
            .unwrap());
    }

    let reopened = LocalFileKvStore::new(&root).await.unwrap();
    assert_eq!(
        reopened.get("text").await.unwrap(),
        Some(KvValue::text("hello"))
    );
    assert_eq!(
        reopened.get("bytes").await.unwrap(),
        Some(KvValue::bytes(BINARY.to_vec()))
    );
    // The TTL is stored as an absolute deadline, so it survives the reopen.
    assert!(reopened.exists("lock").await.unwrap());
    assert!(!reopened
        .exclusive_set("lock", KvValue::text("stolen"), None)
        .await
        .unwrap());

    // Expire the lease by backdating its stored deadline rather than sleeping out a
    // real TTL window: the envelope keeps an absolute epoch-millisecond `expiry_ms`,
    // so a store reopened after the rewrite takes exactly the elapsed-TTL path.
    backdate_local_file_deadline(&root, "lock").await;

    let expired = LocalFileKvStore::new(&root).await.unwrap();
    assert!(!expired.exists("lock").await.unwrap());
    assert!(expired
        .exclusive_set("lock", KvValue::text("reacquired"), None)
        .await
        .unwrap());
    assert_eq!(
        expired.get("lock").await.unwrap(),
        Some(KvValue::text("reacquired"))
    );
}

/// Backdates the stored deadline of `key` in a local-filesystem store.
///
/// `LocalFileKvStore` keeps one JSON envelope per key below `<root>/entries/` and
/// stores its TTL as an absolute epoch-millisecond `expiry_ms`; rewriting that field
/// reproduces the elapsed-TTL path without waiting on a real-time window that
/// parallel test load can stretch past the deadline.
async fn backdate_local_file_deadline(root: &Path, key: &str) {
    let entries = root.join("entries");
    let mut directory = tokio::fs::read_dir(&entries).await.unwrap();
    while let Some(entry) = directory.next_entry().await.unwrap() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        let raw = tokio::fs::read(&path).await.unwrap();
        let mut envelope: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        if envelope["key"] != serde_json::json!(key) {
            continue;
        }
        envelope["expiry_ms"] = serde_json::json!(past_epoch_ms());
        tokio::fs::write(&path, serde_json::to_vec(&envelope).unwrap())
            .await
            .unwrap();
        return;
    }
    panic!("no envelope for key `{key}` below {}", entries.display());
}

/// Epoch milliseconds safely in the past.
fn past_epoch_ms() -> i64 {
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();
    i64::try_from(now_ms).unwrap_or(i64::MAX) - 1_000
}

/// Verifies the whole [`KvStore`] contract against one open store.
pub(crate) async fn assert_kv_contract(store: &dyn KvStore) {
    scalar_round_trip(store).await;
    overwrite_and_delete(store).await;
    exclusive_set_expires(store).await;
    set_clears_ttl(store).await;
    prefix_scans(store).await;
    multi_key_operations(store).await;
    pipeline_ordering(store).await;
}

async fn scalar_round_trip(store: &dyn KvStore) {
    store
        .set("contract:text", KvValue::text("hello"))
        .await
        .unwrap();
    store
        .set("contract:bytes", KvValue::bytes(BINARY.to_vec()))
        .await
        .unwrap();

    assert_eq!(
        store.get("contract:text").await.unwrap(),
        Some(KvValue::text("hello"))
    );
    assert_eq!(
        store.get("contract:bytes").await.unwrap(),
        Some(KvValue::bytes(BINARY.to_vec()))
    );
    assert!(store.exists("contract:text").await.unwrap());
    assert!(store.exists("contract:bytes").await.unwrap());

    assert_eq!(store.get("contract:missing").await.unwrap(), None);
    assert!(!store.exists("contract:missing").await.unwrap());
}

async fn overwrite_and_delete(store: &dyn KvStore) {
    store
        .set("contract:overwrite", KvValue::text("first"))
        .await
        .unwrap();
    store
        .set("contract:overwrite", KvValue::text("second"))
        .await
        .unwrap();
    assert_eq!(
        store.get("contract:overwrite").await.unwrap(),
        Some(KvValue::text("second"))
    );

    store.delete("contract:overwrite").await.unwrap();
    assert_eq!(store.get("contract:overwrite").await.unwrap(), None);
    // Deleting a missing key is a no-op, never an error.
    store.delete("contract:overwrite").await.unwrap();
    store.delete("contract:never-written").await.unwrap();
}

async fn exclusive_set_expires(store: &dyn KvStore) {
    // A lease whose deadline is a minute away stays live, so the refusal and the
    // stored value below cannot race it however the test thread is scheduled.
    assert!(store
        .exclusive_set("contract:lock", KvValue::text("first"), Some(LIVE_TTL))
        .await
        .unwrap());
    assert!(!store
        .exclusive_set("contract:lock", KvValue::text("second"), None)
        .await
        .unwrap());
    assert_eq!(
        store.get("contract:lock").await.unwrap(),
        Some(KvValue::text("first"))
    );
    store.delete("contract:lock").await.unwrap();

    // A short lease is only inspected once it is certainly over: the sleep starts
    // after the write, so nothing here assumes the entry is still live.
    assert!(store
        .exclusive_set("contract:short-lock", KvValue::text("first"), Some(TTL))
        .await
        .unwrap());
    tokio::time::sleep(PAST_TTL).await;
    assert!(!store.exists("contract:short-lock").await.unwrap());
    assert!(store
        .exclusive_set("contract:short-lock", KvValue::text("second"), None)
        .await
        .unwrap());
    assert_eq!(
        store.get("contract:short-lock").await.unwrap(),
        Some(KvValue::text("second"))
    );
    // Without a TTL the reacquired key stays live.
    assert!(!store
        .exclusive_set("contract:short-lock", KvValue::text("third"), None)
        .await
        .unwrap());
}

async fn set_clears_ttl(store: &dyn KvStore) {
    assert!(store
        .exclusive_set("contract:renewed", KvValue::text("leased"), Some(TTL))
        .await
        .unwrap());
    store
        .set("contract:renewed", KvValue::text("owned"))
        .await
        .unwrap();

    // `set` clears the TTL either way, and the sleep starts after the write, so the
    // old deadline has certainly passed when it returns; nothing here assumes the
    // lease is still live in between.
    tokio::time::sleep(PAST_TTL).await;
    assert_eq!(
        store.get("contract:renewed").await.unwrap(),
        Some(KvValue::text("owned"))
    );
}

async fn prefix_scans(store: &dyn KvStore) {
    for key in [
        "contract:prefix:c",
        "contract:prefix:a",
        "contract:prefix:b",
    ] {
        store.set(key, KvValue::text(key)).await.unwrap();
    }
    store
        .set("contract:other", KvValue::text("other"))
        .await
        .unwrap();

    let scanned = store.get_by_prefix("contract:prefix:").await.unwrap();
    let keys: Vec<String> = scanned.keys().cloned().collect();
    assert_eq!(
        keys,
        vec![
            "contract:prefix:a".to_string(),
            "contract:prefix:b".to_string(),
            "contract:prefix:c".to_string()
        ]
    );
    assert_eq!(
        scanned.get("contract:prefix:b"),
        Some(&KvValue::text("contract:prefix:b"))
    );

    let everything = store.get_by_prefix("").await.unwrap();
    assert!(everything.contains_key("contract:other"));
    assert!(everything.contains_key("contract:text"));
    assert!(!everything.contains_key("contract:missing"));

    // An expired entry is not part of a prefix read, and deleting by prefix only
    // counts live entries.
    assert!(store
        .exclusive_set("contract:expiring", KvValue::text("gone"), Some(TTL))
        .await
        .unwrap());
    tokio::time::sleep(PAST_TTL).await;
    assert_eq!(
        store
            .get_by_prefix("contract:expiring")
            .await
            .unwrap()
            .len(),
        0
    );
    assert_eq!(
        store.delete_by_prefix("contract:expiring").await.unwrap(),
        0
    );

    assert_eq!(store.delete_by_prefix("contract:prefix:").await.unwrap(), 3);
    assert_eq!(store.delete_by_prefix("contract:prefix:").await.unwrap(), 0);
    assert!(!store.exists("contract:prefix:a").await.unwrap());
}

async fn multi_key_operations(store: &dyn KvStore) {
    store
        .set("contract:left", KvValue::text("l"))
        .await
        .unwrap();
    store
        .set("contract:right", KvValue::text("r"))
        .await
        .unwrap();

    let keys = vec![
        "contract:left".to_string(),
        "contract:missing".to_string(),
        "contract:left".to_string(),
        "contract:right".to_string(),
    ];
    assert_eq!(
        store.mget(&keys).await.unwrap(),
        vec![
            Some(KvValue::text("l")),
            None,
            Some(KvValue::text("l")),
            Some(KvValue::text("r")),
        ]
    );

    // A duplicate input key is deleted once, so it counts once.
    let doomed = vec![
        "contract:left".to_string(),
        "contract:left".to_string(),
        "contract:right".to_string(),
        "contract:missing".to_string(),
    ];
    assert_eq!(store.batch_delete(&doomed).await.unwrap(), 2);
    assert_eq!(store.batch_delete(&doomed).await.unwrap(), 0);
    assert!(!store.exists("contract:left").await.unwrap());
}

async fn pipeline_ordering(store: &dyn KvStore) {
    let mut pipeline = store.pipeline();
    pipeline.set("contract:pipeline", KvValue::text("value"));
    pipeline.get("contract:pipeline");
    pipeline.exists("contract:pipeline");
    pipeline.get("contract:pipeline-missing");
    pipeline.exists("contract:pipeline-missing");

    assert_eq!(
        pipeline.execute().await.unwrap(),
        vec![
            None,
            Some(KvValue::text("value")),
            Some(KvValue::text("true")),
            None,
            Some(KvValue::text("false")),
        ]
    );
    // The queue is drained, so a second execute performs no operation.
    assert_eq!(pipeline.execute().await.unwrap().len(), 0);
}
