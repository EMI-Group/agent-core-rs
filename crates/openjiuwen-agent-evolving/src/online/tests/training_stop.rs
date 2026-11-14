//! Unit tests of stopping and recovering a training run.
//!
//! Every stop stage — queued, training through the executor, activating, and after
//! a terminal state — is exercised against the crash-recovery matrix, whose
//! authority is the transition table in `super::super::training` and the restart
//! rule behind `SERVICE_RESTARTED`.
use std::path::Path;
use std::sync::Arc;

use super::super::activation::LoraActivation;
use super::super::sample::SampleStatus;
use super::super::store::{keys, OnlineStore};
use super::super::training::{
    RunStage, RunStatus, TrainingRunRecord, TrainingRunnerConfig, SERVICE_RESTARTED,
};
use super::support::*;

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
