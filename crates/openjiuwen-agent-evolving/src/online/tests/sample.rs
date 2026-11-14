//! Unit tests of the sample ledger ([`super::super::sample`]).
//!
//! Every test drives [`SampleLedger`] over [`MemoryOnlineStore`], so ordering is
//! creation order (never a clock) and the assertions cover the documented status
//! machine: first-wins publication, the training fetch, the three transitions and
//! the counters.

use std::sync::Arc;

use serde_json::json;

use super::super::sample::{SampleLedger, SampleRecord, SampleStatus};
use super::super::store::{keys, MemoryOnlineStore, OnlineStore};

type Store = Arc<MemoryOnlineStore>;

fn store() -> Store {
    Arc::new(MemoryOnlineStore::new())
}

fn ledger(store: &Store) -> SampleLedger<Store> {
    SampleLedger::new(store.clone())
}

fn sample(sample_id: &str, reward: f64) -> SampleRecord {
    SampleRecord {
        session_id: "sess-1".to_owned(),
        task_id: "task-1".to_owned(),
        ..SampleRecord::pending(sample_id, "owner", reward, json!({ "turn": 1 }))
    }
}

async fn seed(store: &Store, ids: &[&str]) {
    let records: Vec<SampleRecord> = ids
        .iter()
        .enumerate()
        .map(|(index, sample_id)| sample(sample_id, index as f64 / 10.0))
        .collect();
    let saved = ledger(store).save_once(&records).await.expect("save");
    assert_eq!(saved.len(), ids.len());
}

#[tokio::test]
async fn save_once_assigns_creation_order_and_pending_status() {
    let store = store();
    seed(&store, &["s2", "s1", "s3"]).await;

    let records = ledger(&store)
        .list_by_status(SampleStatus::Pending)
        .await
        .expect("list");
    let ids: Vec<&str> = records
        .iter()
        .map(|record| record.sample_id.as_str())
        .collect();
    assert_eq!(ids, ["s2", "s1", "s3"], "creation order, not id order");
    assert_eq!(
        records
            .iter()
            .map(|record| record.created_seq)
            .collect::<Vec<_>>(),
        [1, 2, 3]
    );
    assert!(records
        .iter()
        .all(|record| record.status == SampleStatus::Pending));
    assert_eq!(
        ledger(&store)
            .count_by_status(SampleStatus::Pending)
            .await
            .expect("count"),
        3
    );
}

#[tokio::test]
async fn save_once_is_first_wins_and_never_resurrects() {
    let store = store();
    seed(&store, &["s1", "s2"]).await;
    // Drain s1 into training, then replay both ids with a new reward and status.
    let claimed = ledger(&store)
        .fetch_and_mark_training(1)
        .await
        .expect("fetch");
    assert_eq!(claimed[0].sample_id, "s1");

    let replay = vec![
        SampleRecord {
            status: SampleStatus::Failed,
            reward: 9.0,
            ..sample("s1", 9.0)
        },
        sample("s2", 9.0),
        sample("s3", 0.5),
    ];
    let saved = ledger(&store).save_once(&replay).await.expect("replay");
    assert_eq!(saved, ["s3"], "only the unseen id is published");

    let existing = ledger(&store).get("s1").await.expect("get").expect("s1");
    assert_eq!(existing.status, SampleStatus::Training, "not resurrected");
    assert!((existing.reward - 0.0).abs() < f64::EPSILON, "reward kept");
    let s2 = ledger(&store).get("s2").await.expect("get").expect("s2");
    assert!((s2.reward - 0.1).abs() < f64::EPSILON);
}

#[tokio::test]
async fn save_once_rejects_blank_ids_and_non_finite_rewards() {
    let store = store();
    let blank = SampleRecord::pending("  ", "owner", 0.0, json!({}));
    let error = ledger(&store).save_once(&[blank]).await.unwrap_err();
    assert!(matches!(
        error,
        super::super::error::OnlineError::InvalidRequest(ref message) if message == "sample_id is required"
    ));

    let nan = SampleRecord::pending("s1", "owner", f64::NAN, json!({}));
    let error = ledger(&store).save_once(&[nan]).await.unwrap_err();
    assert!(matches!(
        error,
        super::super::error::OnlineError::InvalidRequest(ref message)
            if message.contains("reward must be finite")
    ));
    assert_eq!(
        ledger(&store)
            .count_by_status(SampleStatus::Pending)
            .await
            .expect("count"),
        1,
        "only the rejected sample is missing"
    );
}

#[tokio::test]
async fn fetch_and_mark_training_takes_the_oldest_ids_up_to_the_limit() {
    let store = store();
    seed(&store, &["s1", "s2", "s3", "s4"]).await;

    let batch = ledger(&store)
        .fetch_and_mark_training(2)
        .await
        .expect("fetch");
    let ids: Vec<&str> = batch
        .iter()
        .map(|record| record.sample_id.as_str())
        .collect();
    assert_eq!(ids, ["s1", "s2"]);
    assert!(batch
        .iter()
        .all(|record| record.status == SampleStatus::Training));
    assert_eq!(
        ledger(&store)
            .count_by_status(SampleStatus::Training)
            .await
            .expect("count"),
        2
    );
    assert_eq!(
        ledger(&store)
            .count_by_status(SampleStatus::Pending)
            .await
            .expect("count"),
        2
    );

    let second = ledger(&store)
        .fetch_and_mark_training(10)
        .await
        .expect("second fetch");
    let ids: Vec<&str> = second
        .iter()
        .map(|record| record.sample_id.as_str())
        .collect();
    assert_eq!(ids, ["s3", "s4"]);
    assert!(ledger(&store)
        .fetch_and_mark_training(1)
        .await
        .expect("empty fetch")
        .is_empty());
}

#[tokio::test]
async fn transitions_move_an_id_between_exactly_one_index() {
    let store = store();
    seed(&store, &["s1", "s2"]).await;
    let ids: Vec<String> = ledger(&store)
        .fetch_and_mark_training(2)
        .await
        .expect("fetch")
        .into_iter()
        .map(|record| record.sample_id)
        .collect();

    ledger(&store).mark_trained(&ids).await.expect("trained");
    assert_eq!(
        ledger(&store)
            .count_by_status(SampleStatus::Trained)
            .await
            .expect("count"),
        2
    );
    assert_eq!(
        ledger(&store)
            .count_by_status(SampleStatus::Training)
            .await
            .expect("count"),
        0
    );
    // A replay is a no-op, not a second index entry.
    ledger(&store).mark_trained(&ids).await.expect("replay");
    assert_eq!(
        ledger(&store)
            .count_by_status(SampleStatus::Trained)
            .await
            .expect("count"),
        2
    );

    ledger(&store).reset_to_pending(&ids).await.expect("reset");
    assert_eq!(
        ledger(&store)
            .count_by_status(SampleStatus::Pending)
            .await
            .expect("count"),
        2
    );
    ledger(&store).mark_failed(&ids).await.expect("failed");
    assert_eq!(
        ledger(&store)
            .count_by_status(SampleStatus::Failed)
            .await
            .expect("count"),
        2
    );
    // Unknown ids are ignored, as Python's `_update_status` does.
    ledger(&store)
        .mark_trained(&["ghost".to_owned()])
        .await
        .expect("ghost");
}

#[tokio::test]
async fn stats_use_pythons_counter_shape() {
    let store = store();
    seed(&store, &["s1", "s2", "s3", "s4"]).await;
    let claimed = ledger(&store)
        .fetch_and_mark_training(2)
        .await
        .expect("fetch")
        .into_iter()
        .map(|record| record.sample_id)
        .collect::<Vec<_>>();
    ledger(&store)
        .mark_trained(&claimed[..1])
        .await
        .expect("trained");
    ledger(&store)
        .mark_failed(&claimed[1..])
        .await
        .expect("failed");

    let stats = ledger(&store).stats().await.expect("stats");
    assert_eq!(stats.get("total_samples"), Some(&4));
    assert_eq!(stats.get("pending_samples"), Some(&2));
    assert_eq!(stats.get("training_samples"), Some(&0));
    assert_eq!(stats.get("trained_samples"), Some(&1));
    assert_eq!(stats.get("failed_samples"), Some(&1));
}

#[tokio::test]
async fn records_round_trip_through_the_store_and_policy_version_defaults_to_base() {
    let store = store();
    let with_version = SampleRecord::pending(
        "s1",
        "owner",
        0.5,
        json!({ "policy_version": "model-a:v3", "turns": 2 }),
    );
    assert_eq!(with_version.policy_version(), "model-a:v3");
    assert_eq!(sample("s2", 0.0).policy_version(), "base");

    ledger(&store)
        .save_once(&[with_version])
        .await
        .expect("save");
    let stored = ledger(&store).get("s1").await.expect("get").expect("s1");
    assert_eq!(
        stored.payload,
        json!({ "policy_version": "model-a:v3", "turns": 2 })
    );
    let key = keys::sample("s1");
    let raw = store.get(&key).await.expect("read").expect("stored");
    assert_eq!(SampleRecord::from_json(&raw).expect("decode"), stored);
}
