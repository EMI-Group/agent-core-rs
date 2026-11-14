//! Unit tests of claiming and executing a training run.
//!
//! Claim (single flight, sample minimum, batch cap), the full training →
//! activation path, executor and artifact-name failures, checkpoint export and the
//! public record shape live here; the stop/recover matrix is in `training_stop`,
//! and the shared fakes in `support`.
use std::path::Path;
use std::sync::Arc;

use super::super::activation::{NoopLoraActivator, UnsupportedLoraActivator};
use super::super::checkpoint::TrainingCheckpoint;
use super::super::error::OnlineError;
use super::super::executor::{LocalTrainingExecutor, TrainingArtifact};
use super::super::sample::{SampleRecord, SampleStatus};
use super::super::store::{keys, OnlineStore};
use super::super::task::EPOCH_STAMP;
use super::super::training::{RunStage, RunStatus, TrainingRunRecord, TrainingRunnerConfig};
use super::support::*;

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
