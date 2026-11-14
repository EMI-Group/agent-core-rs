//! The online RL state machines over [`RedisOnlineStore`].
//!
//! Every test runs against the in-process `RESP2` server in [`fake_redis`], so the
//! suite is offline: no external `redis-server`, no fixed port, no credentials.
//! The same behaviours proven against [`MemoryOnlineStore`] in the parent module
//! are re-proven here through the real command mapping — primitives, key
//! prefixing, the task lifecycle, the sample ledger's first-wins publication, a
//! full training run with checkpoint export, durability across two store
//! instances, and the typed rejection of TLS/cluster URLs.
use std::sync::Arc;
use std::time::Duration;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_extensions::redis::Client;
use serde_json::json;

use super::super::checkpoint::TrainingCheckpoint;
use super::super::redis_store::{RedisOnlineStore, RedisOnlineStoreOptions};
use super::super::sample::{SampleLedger, SampleStatus};
use super::super::store::{keys, OnlineStore};
use super::super::task::{
    FinishReason, FixedClock, RewardMode, TaskRegistry, TaskSpec, TaskStatus,
};
use super::super::training::{RunStage, RunStatus, TrainingRunner};
use super::fake_redis::FakeRedis;
use super::support::{config, sample, FakeExecutor, Outcome, RecordingActivator, STAMP};

const SESSION: &str = "sess-1";
const TASK: &str = "task-1";

fn spec(task_id: &str, session: &str) -> TaskSpec {
    TaskSpec {
        rl_task_id: task_id.to_owned(),
        agent_session_id: session.to_owned(),
        model_id: "model-a".to_owned(),
        policy_lora_name: "policy-a".to_owned(),
        reward_mode: RewardMode::Terminal,
        policy_model: None,
    }
}

/// The durable key of a domain key under the default prefix.
fn durable(key: &str) -> String {
    format!("openjiuwen:rl:{key}")
}

async fn connect(fake: &FakeRedis) -> Arc<RedisOnlineStore> {
    Arc::new(
        RedisOnlineStore::connect(&fake.url())
            .await
            .expect("connect"),
    )
}

async fn save<S: OnlineStore>(store: S, ids: &[&str]) -> Vec<String> {
    let records: Vec<_> = ids.iter().map(|id| sample(id)).collect();
    SampleLedger::new(store)
        .save_once(&records)
        .await
        .expect("save")
}

// ------------------------------------------------------------ primitives

#[tokio::test]
async fn every_primitive_maps_onto_its_redis_command() {
    let fake = FakeRedis::start().await;
    let store = connect(&fake).await;
    store.ping().await.expect("ping");
    assert!(fake.command_count() > 0);

    assert_eq!(store.get("k").await.expect("get missing"), None);
    store.set("k", "v").await.expect("set");
    assert_eq!(store.get("k").await.expect("get").as_deref(), Some("v"));
    store.set("k", "v2").await.expect("replace");
    assert_eq!(store.get("k").await.expect("get").as_deref(), Some("v2"));

    assert!(store.set_if_absent("fresh", "first").await.expect("nx"));
    assert!(!store.set_if_absent("fresh", "second").await.expect("nx"));
    assert!(!store.set_if_absent("k", "third").await.expect("nx"));
    assert_eq!(
        store.get("fresh").await.expect("get").as_deref(),
        Some("first")
    );

    assert!(store.set_add("s", "b").await.expect("sadd"));
    assert!(store.set_add("s", "a").await.expect("sadd"));
    assert!(!store.set_add("s", "a").await.expect("sadd"));
    // SMEMBERS arrives in hash order; the store sorts it.
    assert_eq!(
        store.set_members("s").await.expect("members"),
        vec!["a".to_owned(), "b".to_owned()]
    );
    assert!(store.set_contains("s", "a").await.expect("sismember"));
    assert!(!store.set_contains("s", "z").await.expect("sismember"));
    assert!(store.set_remove("s", "a").await.expect("srem"));
    assert!(!store.set_remove("s", "a").await.expect("srem"));
    assert_eq!(store.set_members("s").await.expect("members"), ["b"]);

    assert!(store.del("k").await.expect("del"));
    assert!(!store.del("k").await.expect("del"));
    assert_eq!(store.get("k").await.expect("get deleted"), None);
    assert_eq!(
        store.set_members("missing").await.expect("members"),
        Vec::<String>::new()
    );
}

#[tokio::test]
async fn every_key_is_written_under_the_prefix() {
    let fake = FakeRedis::start().await;
    let store = connect(&fake).await;
    assert_eq!(store.prefix(), "openjiuwen:rl");

    store.set(&keys::task(TASK), "{}").await.expect("set");
    store
        .set_add(&keys::captures(TASK), "c1")
        .await
        .expect("sadd");
    assert_eq!(
        fake.state().get(&durable(&keys::task(TASK))).as_deref(),
        Some("{}")
    );
    assert_eq!(
        fake.state().members(&durable(&keys::captures(TASK))),
        ["c1"]
    );
    assert!(fake
        .state()
        .keys()
        .iter()
        .all(|key| key.starts_with("openjiuwen:rl:")));

    // A store built from an existing client keeps its own namespace, so two
    // stores can share one database.
    let client = Client::from_url(&fake.url()).await.expect("client");
    let other = RedisOnlineStore::from_client(client, "team:a:");
    assert_eq!(other.prefix(), "team:a");
    other.set(&keys::task(TASK), "{}").await.expect("set");
    assert_eq!(
        fake.state().get("team:a:rl:v1:task:task-1").as_deref(),
        Some("{}")
    );

    // A blank prefix falls back to the default one.
    let client = Client::from_url(&fake.url()).await.expect("client");
    assert_eq!(
        RedisOnlineStore::from_client(client, "").prefix(),
        "openjiuwen:rl"
    );
}

#[tokio::test]
async fn options_apply_the_prefix_and_the_database() {
    let fake = FakeRedis::start().await;
    let options = RedisOnlineStoreOptions {
        url: fake.url(),
        prefix: "team:b:".to_owned(),
        database: Some(2),
        timeout: Duration::from_secs(5),
    };
    let store = RedisOnlineStore::connect_with_options(&options)
        .await
        .expect("connect");
    assert_eq!(store.prefix(), "team:b");
    store.set("k", "v").await.expect("set");
    assert_eq!(fake.state().get("team:b:k").as_deref(), Some("v"));
}

#[tokio::test]
async fn a_wrong_type_reply_surfaces_as_an_error() {
    let fake = FakeRedis::start().await;
    let store = connect(&fake).await;
    store
        .set_add(&keys::captures(TASK), "c1")
        .await
        .expect("sadd");
    let error = store
        .get(&keys::captures(TASK))
        .await
        .expect_err("a set read as a string");
    assert_eq!(error.kind(), ErrorKind::Validation, "{error}");

    store.set(&keys::task(TASK), "{}").await.expect("set");
    let error = store
        .set_members(&keys::task(TASK))
        .await
        .expect_err("a string read as a set");
    assert_eq!(error.kind(), ErrorKind::Validation, "{error}");
}

#[tokio::test]
async fn tls_and_cluster_urls_are_rejected() {
    for url in [
        "rediss://127.0.0.1:6379/0",
        "redis+cluster://127.0.0.1:6379/0",
        "rediss+cluster://127.0.0.1:6379/0",
    ] {
        let error = RedisOnlineStore::connect(url)
            .await
            .expect_err("unsupported");
        assert_eq!(error.kind(), ErrorKind::Unsupported, "url {url}");
    }
    let tls = RedisOnlineStore::connect("rediss://127.0.0.1:6379/0")
        .await
        .expect_err("tls");
    assert!(tls.to_string().contains("TLS"), "{tls}");
    let cluster = RedisOnlineStore::connect("redis+cluster://127.0.0.1:6379/0")
        .await
        .expect_err("cluster");
    assert!(cluster.to_string().contains("cluster"), "{cluster}");
}

#[tokio::test]
async fn connecting_after_shutdown_surfaces_a_transport_error() {
    let fake = FakeRedis::start().await;
    let url = fake.url();
    fake.shutdown().await;
    let error = RedisOnlineStore::connect(&url)
        .await
        .expect_err("no listener");
    assert_eq!(error.kind(), ErrorKind::Transport, "{error}");
}

// ---------------------------------------------------------- task machine

#[tokio::test]
async fn task_lifecycle_runs_over_redis() {
    let fake = FakeRedis::start().await;
    let store = connect(&fake).await;
    let registry = TaskRegistry::with_clock(store, Arc::new(FixedClock::new(STAMP)));

    let (record, created) = registry.start(&spec(TASK, SESSION)).await.expect("start");
    assert!(created);
    assert_eq!(record.status, TaskStatus::Active);
    assert_eq!(
        fake.state().get(&durable(&keys::task(TASK))),
        Some(record.to_json().expect("json"))
    );
    assert_eq!(
        fake.state()
            .get(&durable(&keys::active_session(SESSION)))
            .as_deref(),
        Some(TASK)
    );
    assert_eq!(
        fake.state().members(&durable(&keys::active_tasks())),
        [TASK]
    );

    // A replay of the same spec adopts the record; another spec conflicts.
    let (replay, created) = registry.start(&spec(TASK, SESSION)).await.expect("replay");
    assert!(!created);
    assert_eq!(replay.created_at, STAMP);
    let conflict = registry
        .start(&spec(TASK, "other-session"))
        .await
        .expect_err("conflict");
    assert_eq!(conflict.code(), "task_conflict");

    // Capture: begin, commit, and the staged sample.
    assert!(registry
        .begin_capture(TASK, "c1", None, "request-1")
        .await
        .expect("begin")
        .is_none());
    assert!(registry
        .commit_capture(TASK, "c1", None, "response-1", &json!({ "id": "c1" }))
        .await
        .expect("commit"));
    assert_eq!(
        fake.state()
            .members(&durable(&keys::committed_captures(TASK))),
        ["c1"]
    );
    assert!(fake
        .state()
        .members(&durable(&keys::open_captures(TASK)))
        .is_empty());
    assert!(fake
        .state()
        .get(&durable(&keys::pending_sample(TASK, "c1")))
        .is_some());

    let finalized = registry
        .finish(TASK, FinishReason::UserStopped)
        .await
        .expect("finish");
    assert_eq!(finalized.status, TaskStatus::Finalized);
    assert_eq!(
        fake.state().get(&durable(&keys::active_session(SESSION))),
        None
    );
    assert!(fake
        .state()
        .members(&durable(&keys::active_tasks()))
        .is_empty());

    // Reward projects onto the committed capture exactly once.
    assert_eq!(registry.record_reward(TASK, 0.25).await.expect("reward"), 1);
    assert_eq!(
        fake.state().get(&durable(&keys::reward(TASK))),
        Some(json!({ "reward": 0.25, "status": "published", "sample_count": 1 }).to_string())
    );
    assert_eq!(
        fake.state()
            .get(&durable(&keys::pending_sample(TASK, "c1"))),
        None
    );
    assert_eq!(registry.record_reward(TASK, 0.25).await.expect("replay"), 1);
    assert_eq!(
        registry
            .finish(TASK, FinishReason::UserStopped)
            .await
            .expect("idempotent finish")
            .finished_at,
        finalized.finished_at
    );
}

// -------------------------------------------------------- sample ledger

#[tokio::test]
async fn the_sample_ledger_is_first_wins_on_redis() {
    let fake = FakeRedis::start().await;
    let store = connect(&fake).await;
    let ledger = SampleLedger::new(store.clone());

    assert_eq!(
        ledger
            .save_once(&[sample("s1"), sample("s2")])
            .await
            .expect("save"),
        ["s1", "s2"]
    );
    assert_eq!(
        fake.state()
            .members(&durable(&keys::samples_by_status("pending"))),
        ["s1", "s2"]
    );

    // A replayed id keeps the record the first writer published.
    let mut replay = sample("s1");
    replay.reward = 0.9;
    assert!(ledger
        .save_once(&[replay])
        .await
        .expect("replay")
        .is_empty());
    let stored = ledger.get("s1").await.expect("get").expect("present");
    assert_eq!(stored.reward, 0.5);

    let batch = ledger.fetch_and_mark_training(10).await.expect("fetch");
    assert_eq!(
        batch
            .iter()
            .map(|record| record.sample_id.clone())
            .collect::<Vec<_>>(),
        ["s1", "s2"]
    );
    assert!(batch
        .iter()
        .all(|record| record.status == SampleStatus::Training));
    assert!(fake
        .state()
        .members(&durable(&keys::samples_by_status("pending")))
        .is_empty());
    assert_eq!(
        fake.state()
            .members(&durable(&keys::samples_by_status("training"))),
        ["s1", "s2"]
    );

    let ids: Vec<String> = batch
        .iter()
        .map(|record| record.sample_id.clone())
        .collect();
    ledger.mark_trained(&ids).await.expect("trained");
    assert_eq!(
        ledger
            .count_by_status(SampleStatus::Trained)
            .await
            .expect("count"),
        2
    );
    assert_eq!(
        fake.state()
            .members(&durable(&keys::samples_by_status("trained"))),
        ["s1", "s2"]
    );
}

// ------------------------------------------------------- training runner

#[tokio::test]
async fn the_training_runner_claims_runs_and_exports_over_redis() {
    let temp = tempfile::tempdir().expect("tempdir");
    let fake = FakeRedis::start().await;
    let store = connect(&fake).await;
    save(store.clone(), &["s1", "s2"]).await;

    let executor = FakeExecutor::new(Outcome::Succeed);
    let activator = RecordingActivator::new(None);
    let runner = TrainingRunner::new(store.clone(), executor, activator.clone(), config())
        .with_clock(Arc::new(FixedClock::new(STAMP)));

    let (run, created, samples) = runner.claim().await.expect("claim");
    assert!(created);
    assert_eq!(run.training_run_id, "run-000000000001");
    assert_eq!(samples.len(), 2);
    assert_eq!(
        fake.state()
            .get(&durable(&keys::active_training_run()))
            .as_deref(),
        Some("run-000000000001")
    );

    // The claim is single flight: a second claim adopts the marker's run.
    let (again, created, nothing) = runner.claim().await.expect("second claim");
    assert!(!created);
    assert_eq!(again.training_run_id, run.training_run_id);
    assert!(nothing.is_empty());

    let done = runner.run(&run.training_run_id).await.expect("run");
    assert_eq!(done.status, RunStatus::Succeeded);
    assert_eq!(done.stage, RunStage::Activating);
    assert_eq!(done.lora_name.as_deref(), Some("model-a:v1"));
    assert_eq!(
        fake.state().get(&durable(&keys::active_training_run())),
        None
    );
    assert_eq!(activator.calls().await.len(), 1);
    assert!(fake
        .state()
        .get(&durable(&keys::training_artifact(&run.training_run_id)))
        .is_some());
    assert_eq!(
        fake.state()
            .members(&durable(&keys::samples_by_status("trained"))),
        ["s1", "s2"]
    );

    let export = runner
        .export_checkpoint(&run.training_run_id, temp.path())
        .await
        .expect("export");
    let checkpoint = TrainingCheckpoint::load_from_path(&export)
        .await
        .expect("load");
    assert_eq!(checkpoint.model_id, "model-a");
    assert_eq!(checkpoint.status, RunStatus::Succeeded);
    assert_eq!(checkpoint.sample_ids, ["s1", "s2"]);
    assert_eq!(
        checkpoint.artifact.expect("artifact descriptor").lora_name,
        "model-a:v1"
    );
}

#[tokio::test]
async fn a_second_instance_reads_the_durable_state() {
    let fake = FakeRedis::start().await;
    let writer = connect(&fake).await;
    let reader = connect(&fake).await;

    let registry = TaskRegistry::with_clock(writer.clone(), Arc::new(FixedClock::new(STAMP)));
    registry.start(&spec(TASK, SESSION)).await.expect("start");
    let seen = TaskRegistry::new(reader.clone())
        .get(TASK)
        .await
        .expect("read task");
    assert_eq!(seen.status, TaskStatus::Active);
    assert_eq!(seen.agent_session_id, SESSION);

    save(writer.clone(), &["s1", "s2"]).await;
    let ledger = SampleLedger::new(reader.clone());
    assert_eq!(ledger.pending_count().await.expect("pending"), 2);
    assert_eq!(
        ledger
            .get("s1")
            .await
            .expect("get")
            .expect("present")
            .user_id,
        "owner"
    );

    // A marker written by a run claimed on one instance is visible on the other.
    let runner = TrainingRunner::new(
        writer.clone(),
        FakeExecutor::new(Outcome::Succeed),
        RecordingActivator::new(None),
        config(),
    )
    .with_clock(Arc::new(FixedClock::new(STAMP)));
    let (run, created, _) = runner.claim().await.expect("claim");
    assert!(created);
    assert_eq!(
        reader
            .get(&keys::active_training_run())
            .await
            .expect("marker")
            .as_deref(),
        Some(run.training_run_id.as_str())
    );
    let claimed = TrainingRunner::new(
        reader.clone(),
        FakeExecutor::new(Outcome::Succeed),
        RecordingActivator::new(None),
        config(),
    )
    .with_clock(Arc::new(FixedClock::new(STAMP)));
    let (adopted, created, _) = claimed.claim().await.expect("adopt");
    assert!(!created);
    assert_eq!(adopted.training_run_id, run.training_run_id);
}
