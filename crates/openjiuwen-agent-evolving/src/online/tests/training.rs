//! Unit tests of the training-run state machine ([`super::super::training`]).
//!
//! The suite drives the runner over [`MemoryOnlineStore`] with a [`FixedClock`],
//! a scripted executor and a recording activator, so every branch of the
//! documented lifecycle — claim, run, stop, recover, export — is deterministic.
//! No test sleeps, reads the clock or starts a wrapper process: the gated
//! executor/activator hold a run inside a stage with a semaphore.

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result as KernelResult};
use serde_json::json;
use tokio::sync::{Barrier, Mutex, Semaphore};

use super::super::error::OnlineError;
use super::super::executor::{
    LocalTrainingExecutor, TrainingArtifact, TrainingExecutor, TrainingRequest,
};
use super::super::sample::{SampleLedger, SampleRecord, SampleStatus};
use super::super::store::{keys, MemoryOnlineStore, OnlineStore};
use super::super::task::{FixedClock, EPOCH_STAMP};
use super::super::training::{
    LoraActivation, LoraActivator, NoopLoraActivator, RunStage, RunStatus, TrainingCheckpoint,
    TrainingRunRecord, TrainingRunner, TrainingRunnerConfig, UnsupportedLoraActivator,
    SERVICE_RESTARTED,
};

type Store = Arc<MemoryOnlineStore>;

const STAMP: &str = "2026-01-01T00:00:00+00:00";

fn store() -> Store {
    Arc::new(MemoryOnlineStore::new())
}

fn sample(sample_id: &str) -> SampleRecord {
    SampleRecord::pending(sample_id, "owner", 0.5, json!({ "policy_version": "base" }))
}

async fn seed(store: &Store, ids: &[&str]) {
    let records: Vec<SampleRecord> = ids.iter().map(|id| sample(id)).collect();
    let saved = SampleLedger::new(store.clone())
        .save_once(&records)
        .await
        .expect("seed");
    assert_eq!(saved.len(), ids.len());
}

async fn status_count(store: &Store, status: SampleStatus) -> usize {
    SampleLedger::new(store.clone())
        .count_by_status(status)
        .await
        .expect("count")
}

async fn active_marker(store: &Store) -> Option<String> {
    store
        .get(&keys::active_training_run())
        .await
        .expect("read marker")
}

fn config() -> TrainingRunnerConfig {
    TrainingRunnerConfig {
        model_id: "model-a".to_owned(),
        min_samples_for_training: 2,
        max_samples_per_run: 4,
        ..TrainingRunnerConfig::default()
    }
}

// --------------------------------------------------------------- fakes

/// A gate that holds an executor or activator inside its call until the test
/// releases it. `enter` blocks the callee; `wait` blocks the test until it
/// arrived; `open` lets the callee continue.
struct Gate {
    entered: Barrier,
    release: Semaphore,
}

impl Gate {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            entered: Barrier::new(2),
            release: Semaphore::new(0),
        })
    }

    async fn enter(&self) {
        self.entered.wait().await;
        let _permit = self.release.acquire().await.expect("release");
    }

    async fn wait(&self) {
        self.entered.wait().await;
    }

    fn open(&self) {
        self.release.add_permits(1);
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Outcome {
    Succeed,
    Named(String),
    Fail(String),
}

struct FakeExecutor {
    outcome: Mutex<Outcome>,
    requests: Mutex<Vec<TrainingRequest>>,
    cancelled: Mutex<Vec<String>>,
    gate: Option<Arc<Gate>>,
}

impl FakeExecutor {
    fn new(outcome: Outcome) -> Arc<Self> {
        Arc::new(Self {
            outcome: Mutex::new(outcome),
            requests: Mutex::new(Vec::new()),
            cancelled: Mutex::new(Vec::new()),
            gate: None,
        })
    }

    fn gated(outcome: Outcome, gate: &Arc<Gate>) -> Arc<Self> {
        Arc::new(Self {
            outcome: Mutex::new(outcome),
            requests: Mutex::new(Vec::new()),
            cancelled: Mutex::new(Vec::new()),
            gate: Some(gate.clone()),
        })
    }

    async fn requests(&self) -> Vec<TrainingRequest> {
        self.requests.lock().await.clone()
    }

    async fn cancelled(&self) -> Vec<String> {
        self.cancelled.lock().await.clone()
    }
}

#[async_trait]
impl TrainingExecutor for FakeExecutor {
    fn name(&self) -> &'static str {
        "fake"
    }

    async fn train_batch(&self, request: &TrainingRequest) -> KernelResult<TrainingArtifact> {
        self.requests.lock().await.push(request.clone());
        if let Some(gate) = &self.gate {
            gate.enter().await;
        }
        let outcome = self.outcome.lock().await.clone();
        match outcome {
            Outcome::Succeed => Ok(TrainingArtifact::new(
                format!("{}:v1", request.model_id),
                "/artifacts/model-a/v1",
            )),
            Outcome::Named(name) => Ok(TrainingArtifact::new(name, "/artifacts/foreign")),
            Outcome::Fail(message) => Err(Error::workflow(message)),
        }
    }

    async fn cancel(&self, training_run_id: &str) -> KernelResult<bool> {
        self.cancelled.lock().await.push(training_run_id.to_owned());
        Ok(true)
    }

    async fn check_health(&self) -> KernelResult<()> {
        Ok(())
    }
}

#[derive(Default)]
struct RecordingActivator {
    calls: Mutex<Vec<LoraActivation>>,
    failure: Mutex<Option<String>>,
    gate: Option<Arc<Gate>>,
}

impl RecordingActivator {
    fn new(failure: Option<String>) -> Arc<Self> {
        Arc::new(Self {
            calls: Mutex::new(Vec::new()),
            failure: Mutex::new(failure),
            gate: None,
        })
    }

    fn gated(failure: Option<String>, gate: &Arc<Gate>) -> Arc<Self> {
        Arc::new(Self {
            calls: Mutex::new(Vec::new()),
            failure: Mutex::new(failure),
            gate: Some(gate.clone()),
        })
    }

    async fn calls(&self) -> Vec<LoraActivation> {
        self.calls.lock().await.clone()
    }
}

#[async_trait]
impl LoraActivator for RecordingActivator {
    async fn activate(&self, request: &LoraActivation) -> KernelResult<()> {
        self.calls.lock().await.push(request.clone());
        if let Some(gate) = &self.gate {
            gate.enter().await;
        }
        match self.failure.lock().await.clone() {
            Some(message) => Err(Error::workflow(message)),
            None => Ok(()),
        }
    }
}

fn runner(
    store: &Store,
    executor: Arc<dyn TrainingExecutor>,
    activator: Arc<dyn LoraActivator>,
    config: TrainingRunnerConfig,
) -> TrainingRunner<Store> {
    TrainingRunner::new(store.clone(), executor, activator, config)
        .with_clock(Arc::new(FixedClock::new(STAMP)))
}

fn local_runner(temp: &Path, store: &Store) -> TrainingRunner<Store> {
    let activator: Arc<dyn LoraActivator> = Arc::new(NoopLoraActivator::new(store.clone()));
    runner(
        store,
        Arc::new(LocalTrainingExecutor::new(temp)),
        activator,
        config(),
    )
}

// --------------------------------------------------------------- claim

#[tokio::test]
async fn claim_needs_the_configured_minimum_of_pending_samples() {
    let store = store();
    seed(&store, &["s1"]).await;
    let runner = local_runner(Path::new("/unused"), &store);

    let error = runner.claim().await.unwrap_err();
    assert!(matches!(
        error,
        OnlineError::InsufficientSamples { have: 1, need: 2 }
    ));
    assert_eq!(status_count(&store, SampleStatus::Pending).await, 1);
    assert_eq!(active_marker(&store).await, None);
    assert_eq!(
        runner.stats().await.expect("stats").get("total_runs"),
        Some(&0)
    );
}

#[tokio::test]
async fn claim_is_single_flight_and_records_the_batch() {
    let store = store();
    seed(&store, &["s1", "s2", "s3"]).await;
    let activator = Arc::new(NoopLoraActivator::new(store.clone()));
    let runner = runner(
        &store,
        FakeExecutor::new(Outcome::Succeed),
        activator,
        config(),
    );

    let (run, created, samples) = runner.claim().await.expect("claim");
    assert!(created);
    assert_eq!(run.training_run_id, "run-000000000001");
    assert_eq!(run.status, RunStatus::Pending);
    assert_eq!(run.stage, RunStage::Queued);
    assert_eq!(run.created_at, STAMP);
    assert_eq!(run.sample_count, 3);
    assert_eq!(run.sample_ids, ["s1", "s2", "s3"]);
    assert_eq!(run.policy_versions.get("base"), Some(&3));
    assert_eq!(run.parent_lora_name(), "base");
    assert_eq!(run.parent_lora_path(), "");
    assert_eq!(samples.len(), 3);
    assert_eq!(status_count(&store, SampleStatus::Training).await, 3);
    assert_eq!(
        active_marker(&store).await.as_deref(),
        Some("run-000000000001")
    );

    let (again, created, samples) = runner.claim().await.expect("second claim");
    assert!(!created);
    assert_eq!(again.training_run_id, run.training_run_id);
    assert_eq!(samples, Vec::<SampleRecord>::new());
    assert_eq!(status_count(&store, SampleStatus::Training).await, 3);
}

#[tokio::test]
async fn claim_caps_the_batch_at_max_samples_per_run() {
    let store = store();
    seed(&store, &["s1", "s2", "s3", "s4", "s5"]).await;
    let runner = runner(
        &store,
        FakeExecutor::new(Outcome::Succeed),
        Arc::new(NoopLoraActivator::new(store.clone())),
        TrainingRunnerConfig {
            min_samples_for_training: 1,
            max_samples_per_run: 3,
            ..config()
        },
    );
    let (run, created, samples) = runner.claim().await.expect("claim");
    assert!(created);
    assert_eq!(run.sample_count, 3);
    assert_eq!(run.sample_ids, ["s1", "s2", "s3"]);
    assert_eq!(samples.len(), 3);
    assert_eq!(status_count(&store, SampleStatus::Pending).await, 2);
}

#[tokio::test]
async fn claim_validates_the_configuration() {
    let store = store();
    seed(&store, &["s1", "s2"]).await;
    let blank = runner(
        &store,
        FakeExecutor::new(Outcome::Succeed),
        Arc::new(UnsupportedLoraActivator::new()),
        TrainingRunnerConfig {
            model_id: "  ".to_owned(),
            ..config()
        },
    );
    let error = blank.claim().await.unwrap_err();
    assert!(matches!(
        error,
        OnlineError::InvalidRequest(ref message) if message == "model_id is required"
    ));

    let inverted = runner(
        &store,
        FakeExecutor::new(Outcome::Succeed),
        Arc::new(UnsupportedLoraActivator::new()),
        TrainingRunnerConfig {
            min_samples_for_training: 4,
            max_samples_per_run: 2,
            ..config()
        },
    );
    let error = inverted.claim().await.unwrap_err();
    assert!(matches!(
        error,
        OnlineError::InvalidRequest(ref message)
            if message == "max_samples_per_run must be >= min_samples_for_training"
    ));
    assert_eq!(status_count(&store, SampleStatus::Pending).await, 2);
}

// ----------------------------------------------------------------- run

#[tokio::test]
async fn a_full_run_trains_activates_and_succeeds() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = store();
    seed(&store, &["s1", "s2", "s3"]).await;
    let activator = Arc::new(NoopLoraActivator::new(store.clone()));
    let runner = runner(
        &store,
        Arc::new(LocalTrainingExecutor::new(temp.path())),
        activator.clone(),
        config(),
    );

    let (queued, created, _) = runner.claim().await.expect("claim");
    assert!(created);
    let done = runner.run(&queued.training_run_id).await.expect("run");

    assert_eq!(done.status, RunStatus::Succeeded);
    assert_eq!(done.stage, RunStage::Activating);
    assert_eq!(done.lora_name.as_deref(), Some("model-a:v1"));
    assert!(done
        .lora_path
        .as_deref()
        .expect("path")
        .ends_with("v1.json"));
    assert_eq!(done.started_at.as_deref(), Some(STAMP));
    assert_eq!(done.finished_at.as_deref(), Some(STAMP));
    assert_eq!(done.failure_reason, None);
    assert_eq!(status_count(&store, SampleStatus::Trained).await, 3);
    assert_eq!(active_marker(&store).await, None);

    // The artifact descriptor is durable and the activation was recorded.
    let stored = store
        .get(&keys::training_artifact(&done.training_run_id))
        .await
        .expect("read artifact")
        .expect("artifact record");
    let artifact: TrainingArtifact = serde_json::from_str(&stored).expect("decode");
    assert_eq!(artifact.lora_name, "model-a:v1");
    assert_eq!(artifact.metrics.get("sample_count"), Some(&3.0));

    let activation = activator
        .recorded(&done.training_run_id)
        .await
        .expect("read activation")
        .expect("activation");
    assert_eq!(activation.base_model, "model-a");
    assert_eq!(activation.lora_name, "model-a:v1");
    assert_eq!(activation.expected_lora_name, "base");
    assert_eq!(activation.training_run_id, done.training_run_id);

    // Replaying `run` is idempotent.
    assert_eq!(
        runner.run(&done.training_run_id).await.expect("replay"),
        done
    );

    let stats = runner.stats().await.expect("stats");
    assert_eq!(stats.get("total_runs"), Some(&1));
    assert_eq!(stats.get("succeeded_runs"), Some(&1));
    assert_eq!(stats.get("trained_samples"), Some(&3));
    assert_eq!(stats.get("total_samples"), Some(&3));
}

#[tokio::test]
async fn the_executor_receives_the_parent_policy_and_the_training_config() {
    let store = store();
    seed(&store, &["s1", "s2"]).await;
    let executor = FakeExecutor::new(Outcome::Named("model-a:v3".to_owned()));
    let runner = runner(
        &store,
        executor.clone(),
        RecordingActivator::new(None),
        TrainingRunnerConfig {
            batch_size: 16,
            ..config()
        },
    )
    .with_parent_policy("model-a:v2", "/lora/model-a/v2");

    let (queued, _, _) = runner.claim().await.expect("claim");
    assert_eq!(queued.parent_lora_name(), "model-a:v2");
    assert_eq!(queued.parent_lora_path(), "/lora/model-a/v2");

    let done = runner.run(&queued.training_run_id).await.expect("run");
    assert_eq!(done.lora_name.as_deref(), Some("model-a:v3"));

    let requests = executor.requests().await;
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    assert_eq!(request.training_run_id, queued.training_run_id);
    assert_eq!(request.model_id, "model-a");
    assert_eq!(request.samples.len(), 2);
    assert_eq!(request.init_lora_name.as_deref(), Some("model-a:v2"));
    assert_eq!(request.init_lora_path.as_deref(), Some("/lora/model-a/v2"));
    assert_eq!(request.config.train_batch_size, 16);
    assert_eq!(request.output_dir, None);
    // A parent policy is not an artifact name of this model, so the local
    // version rule is not involved: the executor names the artifact.
    assert_eq!(done.status, RunStatus::Succeeded);
}

#[tokio::test]
async fn an_executor_failure_restores_the_samples_and_fails_the_run() {
    let store = store();
    seed(&store, &["s1", "s2"]).await;
    let runner = runner(
        &store,
        FakeExecutor::new(Outcome::Fail("PPO worker exploded".to_owned())),
        RecordingActivator::new(None),
        config(),
    );

    let (queued, _, _) = runner.claim().await.expect("claim");
    let failed = runner.run(&queued.training_run_id).await.expect("run");

    assert_eq!(failed.status, RunStatus::Failed);
    assert_eq!(failed.stage, RunStage::Training);
    assert_eq!(failed.finished_at.as_deref(), Some(STAMP));
    assert!(failed
        .failure_reason
        .as_deref()
        .expect("reason")
        .contains("PPO worker exploded"));
    assert_eq!(failed.lora_name, None);
    assert_eq!(status_count(&store, SampleStatus::Pending).await, 2);
    assert_eq!(status_count(&store, SampleStatus::Training).await, 0);
    assert_eq!(active_marker(&store).await, None);
    assert_eq!(
        runner.stats().await.expect("stats").get("failed_runs"),
        Some(&1)
    );
}

#[tokio::test]
async fn an_artifact_with_a_foreign_name_fails_the_run() {
    let store = store();
    seed(&store, &["s1", "s2"]).await;
    let runner = runner(
        &store,
        FakeExecutor::new(Outcome::Named("other-model:v1".to_owned())),
        RecordingActivator::new(None),
        config(),
    );

    let (queued, _, _) = runner.claim().await.expect("claim");
    let failed = runner.run(&queued.training_run_id).await.expect("run");
    assert_eq!(failed.status, RunStatus::Failed);
    assert!(failed
        .failure_reason
        .as_deref()
        .expect("reason")
        .contains("PPO artifact must use '<model_id>:vN' name"));
    assert_eq!(failed.lora_name, None);
    assert_eq!(status_count(&store, SampleStatus::Pending).await, 2);
}

#[tokio::test]
async fn an_activation_failure_keeps_the_trained_samples() {
    let store = store();
    seed(&store, &["s1", "s2"]).await;
    let runner = runner(
        &store,
        FakeExecutor::new(Outcome::Succeed),
        Arc::new(UnsupportedLoraActivator::new()),
        config(),
    );

    let (queued, _, _) = runner.claim().await.expect("claim");
    let failed = runner.run(&queued.training_run_id).await.expect("run");
    assert_eq!(failed.status, RunStatus::Failed);
    assert_eq!(failed.stage, RunStage::Activating);
    assert_eq!(failed.lora_name.as_deref(), Some("model-a:v1"));
    let reason = failed.failure_reason.as_deref().expect("reason");
    assert!(reason.contains("AIGW"), "{reason}");
    assert!(reason.contains("unsupported"), "{reason}");
    assert_eq!(status_count(&store, SampleStatus::Trained).await, 2);
    assert_eq!(active_marker(&store).await, None);
}

#[tokio::test]
async fn a_second_execution_of_a_running_run_conflicts() {
    let store = store();
    seed(&store, &["s1", "s2"]).await;
    let gate = Gate::new();
    let runner = Arc::new(runner(
        &store,
        FakeExecutor::gated(Outcome::Succeed, &gate),
        RecordingActivator::new(None),
        config(),
    ));
    let (queued, _, _) = runner.claim().await.expect("claim");

    let in_flight = tokio::spawn({
        let runner = runner.clone();
        let training_run_id = queued.training_run_id.clone();
        async move { runner.run(&training_run_id).await }
    });
    gate.wait().await;

    let error = runner.run(&queued.training_run_id).await.unwrap_err();
    assert!(matches!(
        error,
        OnlineError::TaskConflict(ref message) if message == "training run is not queued: run-000000000001"
    ));

    gate.open();
    let done = in_flight.await.expect("join").expect("run");
    assert_eq!(done.status, RunStatus::Succeeded);
}

#[tokio::test]
async fn get_reports_an_unknown_run() {
    let store = store();
    let runner = local_runner(Path::new("/unused"), &store);
    let error = runner.get("run-ffffffff").await.unwrap_err();
    assert!(matches!(error, OnlineError::TrainingRunNotFound(_)));
    assert_eq!(error.code(), "training_run_not_found");
    assert_eq!(error.http_status(), 404);
    assert!(error.to_string().contains("run-ffffffff"));
}

// ---------------------------------------------------------------- stop

#[tokio::test]
async fn stop_while_queued_cancels_and_requeues_the_samples() {
    let store = store();
    seed(&store, &["s1", "s2"]).await;
    let executor = FakeExecutor::new(Outcome::Succeed);
    let runner = runner(
        &store,
        executor.clone(),
        RecordingActivator::new(None),
        config(),
    );

    let (queued, _, _) = runner.claim().await.expect("claim");
    let stopped = runner.stop(&queued.training_run_id).await.expect("stop");
    assert_eq!(stopped.status, RunStatus::Canceled);
    assert_eq!(stopped.stage, RunStage::Queued);
    assert_eq!(stopped.finished_at.as_deref(), Some(STAMP));
    assert_eq!(status_count(&store, SampleStatus::Pending).await, 2);
    assert_eq!(
        executor.cancelled().await,
        Vec::<String>::new(),
        "nothing was running"
    );
    assert_eq!(active_marker(&store).await, None);

    // A terminal run is never executed again.
    assert_eq!(
        runner.run(&queued.training_run_id).await.expect("run"),
        stopped
    );
    // ... nor stopped twice.
    assert_eq!(
        runner.stop(&queued.training_run_id).await.expect("stop"),
        stopped
    );
}

#[tokio::test]
async fn stop_while_training_cancels_through_the_executor() {
    let store = store();
    seed(&store, &["s1", "s2"]).await;
    let gate = Gate::new();
    let executor = FakeExecutor::gated(Outcome::Succeed, &gate);
    let runner = Arc::new(runner(
        &store,
        executor.clone(),
        RecordingActivator::new(None),
        config(),
    ));
    let (queued, _, _) = runner.claim().await.expect("claim");

    let in_flight = tokio::spawn({
        let runner = runner.clone();
        let training_run_id = queued.training_run_id.clone();
        async move { runner.run(&training_run_id).await }
    });
    gate.wait().await;

    let stopped = runner.stop(&queued.training_run_id).await.expect("stop");
    assert_eq!(stopped.status, RunStatus::Canceled);
    assert_eq!(stopped.stage, RunStage::Training);
    assert_eq!(
        executor.cancelled().await,
        vec![queued.training_run_id.clone()]
    );
    assert_eq!(status_count(&store, SampleStatus::Pending).await, 2);

    // The in-flight run sees the terminal record and does not overwrite it.
    gate.open();
    let settled = in_flight.await.expect("join").expect("run");
    assert_eq!(settled.status, RunStatus::Canceled);
    assert_eq!(status_count(&store, SampleStatus::Pending).await, 2);
    assert_eq!(status_count(&store, SampleStatus::Trained).await, 0);
}

#[tokio::test]
async fn stop_while_activating_keeps_the_samples_trained() {
    let store = store();
    seed(&store, &["s1", "s2"]).await;
    let gate = Gate::new();
    let activator = RecordingActivator::gated(None, &gate);
    let runner = Arc::new(runner(
        &store,
        FakeExecutor::new(Outcome::Succeed),
        activator,
        config(),
    ));
    let (queued, _, _) = runner.claim().await.expect("claim");

    let in_flight = tokio::spawn({
        let runner = runner.clone();
        let training_run_id = queued.training_run_id.clone();
        async move { runner.run(&training_run_id).await }
    });
    gate.wait().await;
    assert_eq!(status_count(&store, SampleStatus::Trained).await, 2);

    let stopped = runner.stop(&queued.training_run_id).await.expect("stop");
    assert_eq!(stopped.status, RunStatus::Canceled);
    assert_eq!(stopped.stage, RunStage::Activating);
    assert_eq!(stopped.lora_name.as_deref(), Some("model-a:v1"));

    gate.open();
    let settled = in_flight.await.expect("join").expect("run");
    assert_eq!(settled.status, RunStatus::Canceled);
    assert_eq!(status_count(&store, SampleStatus::Trained).await, 2);
    assert_eq!(active_marker(&store).await, None);
}

#[tokio::test]
async fn stop_after_a_terminal_run_returns_it_unchanged() {
    let store = store();
    seed(&store, &["s1", "s2"]).await;
    let executor = FakeExecutor::new(Outcome::Succeed);
    let runner = runner(
        &store,
        executor.clone(),
        RecordingActivator::new(None),
        config(),
    );
    let (queued, _, _) = runner.claim().await.expect("claim");
    let done = runner.run(&queued.training_run_id).await.expect("run");
    assert_eq!(done.status, RunStatus::Succeeded);

    assert_eq!(
        runner.stop(&queued.training_run_id).await.expect("stop"),
        done
    );
    assert_eq!(executor.cancelled().await, Vec::<String>::new());
    assert_eq!(status_count(&store, SampleStatus::Trained).await, 2);
}

// ------------------------------------------------------------- recover

#[tokio::test]
async fn recover_fails_a_queued_run_with_the_restart_rule() {
    let store = store();
    seed(&store, &["s1", "s2"]).await;
    let runner = local_runner(Path::new("/unused"), &store);
    let (queued, _, _) = runner.claim().await.expect("claim");

    let recovered = runner.recover().await.expect("recover");
    assert_eq!(recovered.len(), 1);
    assert_eq!(recovered[0].training_run_id, queued.training_run_id);
    assert_eq!(recovered[0].status, RunStatus::Failed);
    assert_eq!(
        recovered[0].failure_reason.as_deref(),
        Some(SERVICE_RESTARTED)
    );
    assert_eq!(status_count(&store, SampleStatus::Pending).await, 2);
    assert_eq!(active_marker(&store).await, None);

    // A second recovery is a no-op: the run is terminal.
    assert_eq!(
        runner.recover().await.expect("recover"),
        Vec::<TrainingRunRecord>::new()
    );
    assert_eq!(
        local_runner(Path::new("/unused"), &store)
            .recover()
            .await
            .expect("recover"),
        Vec::<TrainingRunRecord>::new()
    );
}

#[tokio::test]
async fn recover_fails_a_training_run_and_the_in_flight_run_defers() {
    let store = store();
    seed(&store, &["s1", "s2"]).await;
    let gate = Gate::new();
    let executor = FakeExecutor::gated(Outcome::Succeed, &gate);
    let runner = Arc::new(runner(
        &store,
        executor,
        RecordingActivator::new(None),
        config(),
    ));
    let (queued, _, _) = runner.claim().await.expect("claim");

    let in_flight = tokio::spawn({
        let runner = runner.clone();
        let training_run_id = queued.training_run_id.clone();
        async move { runner.run(&training_run_id).await }
    });
    gate.wait().await;

    let recovered = runner.recover().await.expect("recover");
    assert_eq!(recovered.len(), 1);
    assert_eq!(recovered[0].status, RunStatus::Failed);
    assert_eq!(
        recovered[0].failure_reason.as_deref(),
        Some(SERVICE_RESTARTED)
    );
    assert_eq!(status_count(&store, SampleStatus::Pending).await, 2);

    gate.open();
    let settled = in_flight.await.expect("join").expect("run");
    assert_eq!(settled.status, RunStatus::Failed);
    assert_eq!(status_count(&store, SampleStatus::Training).await, 0);
    assert_eq!(status_count(&store, SampleStatus::Pending).await, 2);
}

#[tokio::test]
async fn recover_activates_an_activating_run() {
    let store = store();
    seed(&store, &["s1", "s2"]).await;
    let gate = Gate::new();
    let in_process = RecordingActivator::gated(None, &gate);
    let first = Arc::new(runner(
        &store,
        FakeExecutor::new(Outcome::Succeed),
        in_process.clone(),
        config(),
    ));
    let (queued, _, _) = first.claim().await.expect("claim");

    let in_flight = tokio::spawn({
        let runner = first.clone();
        let training_run_id = queued.training_run_id.clone();
        async move { runner.run(&training_run_id).await }
    });
    gate.wait().await;

    // A restarted process recovers the persisted activating run itself.
    let restarted_activator = RecordingActivator::new(None);
    let restarted = runner(
        &store,
        FakeExecutor::new(Outcome::Succeed),
        restarted_activator.clone(),
        config(),
    );
    let recovered = restarted.recover().await.expect("recover");
    assert_eq!(recovered.len(), 1);
    assert_eq!(recovered[0].status, RunStatus::Succeeded);
    assert_eq!(recovered[0].stage, RunStage::Activating);
    assert_eq!(recovered[0].lora_name.as_deref(), Some("model-a:v1"));
    assert_eq!(restarted_activator.calls().await.len(), 1);
    assert_eq!(status_count(&store, SampleStatus::Trained).await, 2);
    assert_eq!(active_marker(&store).await, None);

    // The original in-process run settles on the recovered record.
    gate.open();
    let settled = in_flight.await.expect("join").expect("run");
    assert_eq!(settled.status, RunStatus::Succeeded);
    assert_eq!(in_process.calls().await.len(), 1);
}

#[tokio::test]
async fn recover_succeeds_an_activating_run_when_auto_activation_is_off() {
    let store = store();
    seed(&store, &["s1", "s2"]).await;
    let gate = Gate::new();
    let in_process = RecordingActivator::gated(None, &gate);
    let first = Arc::new(runner(
        &store,
        FakeExecutor::new(Outcome::Succeed),
        in_process,
        config(),
    ));
    let (queued, _, _) = first.claim().await.expect("claim");

    let in_flight = tokio::spawn({
        let runner = first.clone();
        let training_run_id = queued.training_run_id.clone();
        async move { runner.run(&training_run_id).await }
    });
    gate.wait().await;

    let restarted_activator = RecordingActivator::new(None);
    let restarted = runner(
        &store,
        FakeExecutor::new(Outcome::Succeed),
        restarted_activator.clone(),
        TrainingRunnerConfig {
            auto_activate_lora: false,
            ..config()
        },
    );
    let recovered = restarted.recover().await.expect("recover");
    assert_eq!(recovered.len(), 1);
    assert_eq!(recovered[0].status, RunStatus::Succeeded);
    assert_eq!(
        restarted_activator.calls().await,
        Vec::<LoraActivation>::new(),
        "auto-activation is off"
    );

    gate.open();
    let settled = in_flight.await.expect("join").expect("run");
    assert_eq!(settled.status, RunStatus::Succeeded);
}

#[tokio::test]
async fn recover_with_auto_activation_off_never_calls_the_activator() {
    let store = store();
    seed(&store, &["s1", "s2"]).await;
    let activator = RecordingActivator::new(None);
    let runner = runner(
        &store,
        FakeExecutor::new(Outcome::Succeed),
        activator.clone(),
        TrainingRunnerConfig {
            auto_activate_lora: false,
            ..config()
        },
    );
    let (queued, _, _) = runner.claim().await.expect("claim");
    let done = runner.run(&queued.training_run_id).await.expect("run");
    assert_eq!(done.status, RunStatus::Succeeded);
    assert_eq!(done.lora_name.as_deref(), Some("model-a:v1"));
    assert_eq!(activator.calls().await, Vec::<LoraActivation>::new());
    assert_eq!(status_count(&store, SampleStatus::Trained).await, 2);
}

#[tokio::test]
async fn recover_clears_a_stale_active_marker() {
    let store = store();
    let runner = local_runner(Path::new("/unused"), &store);
    store
        .set(&keys::active_training_run(), "run-00000000dead")
        .await
        .expect("marker");
    assert!(runner.active_run().await.expect("active").is_none());
    assert_eq!(active_marker(&store).await, None);
}

// ------------------------------------------------------------- export

#[tokio::test]
async fn export_checkpoint_round_trips_the_run_and_its_artifact() {
    let temp = tempfile::tempdir().expect("tempdir");
    let exports = temp.path().join("exports");
    let store = store();
    seed(&store, &["s1", "s2"]).await;
    let runner = local_runner(temp.path(), &store);

    let (queued, _, _) = runner.claim().await.expect("claim");
    let queued_export = runner
        .export_checkpoint(&queued.training_run_id, &exports)
        .await
        .expect("queued export");
    let checkpoint = TrainingCheckpoint::load_from_path(&queued_export)
        .await
        .expect("load queued");
    assert_eq!(checkpoint.status, RunStatus::Pending);
    assert_eq!(checkpoint.stage, RunStage::Queued);
    assert_eq!(checkpoint.sample_ids, ["s1", "s2"]);
    assert_eq!(checkpoint.artifact, None);

    let done = runner.run(&queued.training_run_id).await.expect("run");
    let path = runner
        .export_checkpoint(&done.training_run_id, &exports)
        .await
        .expect("export");
    assert_eq!(
        path.file_name().expect("name").to_string_lossy(),
        format!("training-run-{}.json", done.training_run_id)
    );
    let checkpoint = TrainingCheckpoint::load_from_path(&path)
        .await
        .expect("load");
    assert_eq!(checkpoint.training_run_id, done.training_run_id);
    assert_eq!(checkpoint.model_id, "model-a");
    assert_eq!(checkpoint.status, RunStatus::Succeeded);
    assert_eq!(checkpoint.stage, RunStage::Activating);
    assert_eq!(checkpoint.sample_count, 2);
    assert_eq!(checkpoint.sample_ids, done.sample_ids);
    assert_eq!(checkpoint.policy_versions, done.policy_versions);
    assert_eq!(checkpoint.lora_name.as_deref(), Some("model-a:v1"));
    let artifact = checkpoint.artifact.expect("artifact descriptor");
    assert_eq!(artifact.lora_name, "model-a:v1");
    assert_eq!(artifact.metrics.get("sample_count"), Some(&2.0));
    // The export is the run record plus the artifact: nothing was dropped.
    assert_eq!(checkpoint.failure_reason, None);
    assert_eq!(checkpoint.finished_at.as_deref(), Some(STAMP));
}

#[tokio::test]
async fn export_checkpoint_of_an_unknown_run_is_not_found() {
    let temp = tempfile::tempdir().expect("tempdir");
    let store = store();
    let runner = local_runner(temp.path(), &store);
    let error = runner
        .export_checkpoint("run-00000000dead", temp.path())
        .await
        .unwrap_err();
    assert!(matches!(error, OnlineError::TrainingRunNotFound(_)));
}

// ------------------------------------------------------------ record shape

#[test]
fn the_public_record_omits_the_recovery_fields() {
    let run = TrainingRunRecord {
        training_run_id: "run-000000000001".to_owned(),
        status: RunStatus::Succeeded,
        stage: RunStage::Activating,
        sample_count: 2,
        policy_versions: [("base".to_owned(), 2)].into_iter().collect(),
        created_at: STAMP.to_owned(),
        started_at: Some(STAMP.to_owned()),
        finished_at: Some(STAMP.to_owned()),
        lora_name: Some("model-a:v1".to_owned()),
        lora_path: Some("/artifacts/v1.json".to_owned()),
        failure_reason: None,
        sample_ids: vec!["s1".to_owned(), "s2".to_owned()],
        parent_lora_name: Some("base".to_owned()),
        parent_lora_path: Some(String::new()),
    };

    let public: serde_json::Value =
        serde_json::from_str(&run.to_public_json().expect("public json")).expect("decode");
    for internal in ["sample_ids", "parent_lora_name", "parent_lora_path"] {
        assert!(public.get(internal).is_none(), "{internal} leaked");
    }
    assert_eq!(public["status"], "succeeded");
    assert_eq!(public["stage"], "activating");
    assert_eq!(public["sample_count"], 2);

    // The durable form keeps everything and round-trips.
    let durable = run.to_json().expect("durable json");
    assert_eq!(TrainingRunRecord::from_json(&durable).expect("decode"), run);
    // The public shape parses too, with the recovery fields defaulted.
    let from_public =
        TrainingRunRecord::from_json(&run.to_public_json().expect("public")).expect("decode");
    assert_eq!(from_public, run.public());
    assert_eq!(run.public().sample_ids, Vec::<String>::new());
}

#[test]
fn status_and_stage_literals_match_python() {
    assert_eq!(RunStatus::Pending.as_str(), "pending");
    assert_eq!(RunStatus::Running.as_str(), "running");
    assert_eq!(RunStatus::Succeeded.as_str(), "succeeded");
    assert_eq!(RunStatus::Failed.as_str(), "failed");
    assert_eq!(RunStatus::Canceled.as_str(), "canceled");
    assert_eq!(RunStage::Queued.as_str(), "queued");
    assert_eq!(RunStage::Training.as_str(), "training");
    assert_eq!(RunStage::Activating.as_str(), "activating");
    assert!(!RunStatus::Pending.is_terminal());
    assert!(RunStatus::Succeeded.is_terminal());
    assert_eq!(EPOCH_STAMP, "1970-01-01T00:00:00+00:00");
}
