//! Shared fakes and helpers of the training-run suites.
//!
//! Both `training` (claim/run/export) and `training_stop` (stop/recover) drive the
//! runner over a `MemoryOnlineStore` with a `FixedClock`, a scripted executor and
//! a recording activator, so every branch of the documented lifecycle is
//! deterministic. No test sleeps, reads the clock or starts a wrapper process: the
//! gated executor/activator hold a run inside a stage with a semaphore.
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result as KernelResult};
use serde_json::json;
use tokio::sync::{Barrier, Mutex, Semaphore};

use super::super::activation::{LoraActivation, LoraActivator, NoopLoraActivator};
use super::super::executor::{
    LocalTrainingExecutor, TrainingArtifact, TrainingExecutor, TrainingRequest,
};
use super::super::sample::{SampleLedger, SampleRecord, SampleStatus};
use super::super::store::{keys, MemoryOnlineStore, OnlineStore};
use super::super::task::FixedClock;
use super::super::training::{TrainingRunner, TrainingRunnerConfig};

pub(super) type Store = Arc<MemoryOnlineStore>;

pub(super) const STAMP: &str = "2026-01-01T00:00:00+00:00";

pub(super) fn store() -> Store {
    Arc::new(MemoryOnlineStore::new())
}

pub(super) fn sample(sample_id: &str) -> SampleRecord {
    SampleRecord::pending(sample_id, "owner", 0.5, json!({ "policy_version": "base" }))
}

pub(super) async fn seed(store: &Store, ids: &[&str]) {
    let records: Vec<SampleRecord> = ids.iter().map(|id| sample(id)).collect();
    let saved = SampleLedger::new(store.clone())
        .save_once(&records)
        .await
        .expect("seed");
    assert_eq!(saved.len(), ids.len());
}

pub(super) async fn status_count(store: &Store, status: SampleStatus) -> usize {
    SampleLedger::new(store.clone())
        .count_by_status(status)
        .await
        .expect("count")
}

pub(super) async fn active_marker(store: &Store) -> Option<String> {
    store
        .get(&keys::active_training_run())
        .await
        .expect("read marker")
}

pub(super) fn config() -> TrainingRunnerConfig {
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
pub(super) struct Gate {
    entered: Barrier,
    release: Semaphore,
}

impl Gate {
    pub(super) fn new() -> Arc<Self> {
        Arc::new(Self {
            entered: Barrier::new(2),
            release: Semaphore::new(0),
        })
    }

    pub(super) async fn enter(&self) {
        self.entered.wait().await;
        let _permit = self.release.acquire().await.expect("release");
    }

    pub(super) async fn wait(&self) {
        self.entered.wait().await;
    }

    pub(super) fn open(&self) {
        self.release.add_permits(1);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum Outcome {
    Succeed,
    Named(String),
    Fail(String),
}

pub(super) struct FakeExecutor {
    outcome: Mutex<Outcome>,
    requests: Mutex<Vec<TrainingRequest>>,
    cancelled: Mutex<Vec<String>>,
    gate: Option<Arc<Gate>>,
}

impl FakeExecutor {
    pub(super) fn new(outcome: Outcome) -> Arc<Self> {
        Arc::new(Self {
            outcome: Mutex::new(outcome),
            requests: Mutex::new(Vec::new()),
            cancelled: Mutex::new(Vec::new()),
            gate: None,
        })
    }

    pub(super) fn gated(outcome: Outcome, gate: &Arc<Gate>) -> Arc<Self> {
        Arc::new(Self {
            outcome: Mutex::new(outcome),
            requests: Mutex::new(Vec::new()),
            cancelled: Mutex::new(Vec::new()),
            gate: Some(gate.clone()),
        })
    }

    pub(super) async fn requests(&self) -> Vec<TrainingRequest> {
        self.requests.lock().await.clone()
    }

    pub(super) async fn cancelled(&self) -> Vec<String> {
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
pub(super) struct RecordingActivator {
    calls: Mutex<Vec<LoraActivation>>,
    failure: Mutex<Option<String>>,
    gate: Option<Arc<Gate>>,
}

impl RecordingActivator {
    pub(super) fn new(failure: Option<String>) -> Arc<Self> {
        Arc::new(Self {
            calls: Mutex::new(Vec::new()),
            failure: Mutex::new(failure),
            gate: None,
        })
    }

    pub(super) fn gated(failure: Option<String>, gate: &Arc<Gate>) -> Arc<Self> {
        Arc::new(Self {
            calls: Mutex::new(Vec::new()),
            failure: Mutex::new(failure),
            gate: Some(gate.clone()),
        })
    }

    pub(super) async fn calls(&self) -> Vec<LoraActivation> {
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

pub(super) fn runner(
    store: &Store,
    executor: Arc<dyn TrainingExecutor>,
    activator: Arc<dyn LoraActivator>,
    config: TrainingRunnerConfig,
) -> TrainingRunner<Store> {
    TrainingRunner::new(store.clone(), executor, activator, config)
        .with_clock(Arc::new(FixedClock::new(STAMP)))
}

pub(super) fn local_runner(temp: &Path, store: &Store) -> TrainingRunner<Store> {
    let activator: Arc<dyn LoraActivator> = Arc::new(NoopLoraActivator::new(store.clone()));
    runner(
        store,
        Arc::new(LocalTrainingExecutor::new(temp)),
        activator,
        config(),
    )
}
