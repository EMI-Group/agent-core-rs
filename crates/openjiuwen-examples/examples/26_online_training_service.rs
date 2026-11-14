//! `26_online_training_service` — the online (RL) service state machine, driven
//! end to end over an in-memory store.
//!
//! Python counterpart: `openjiuwen/agent_evolving/agent_rl/online/` — the task
//! registry and capture pipeline (`task_registry.py`, `capture_pipeline.py`), the
//! trajectory ledger (`storage/trajectory_store.py`), the training runner
//! (`online/training_runner.py`) and its two seams (`abstract/trainer.py`,
//! `scheduler/ppo_executor.py`, `storage/lora_repo.py`). The port exposes the same
//! machine as `openjiuwen::agent_evolving::online` over six store primitives; this
//! example drives it with no HTTP layer, no Redis, no GPU and no model:
//!
//! 1. task machine — `start`/replay, the active-session index, `finish`;
//! 2. capture — `begin_capture`, `commit_capture` stages raw JSON, `discard_capture`;
//! 3. ledger — `SampleLedger::save_once` first-wins publication and `stats`;
//! 4. training — `claim` a batch, `run` it through `LocalTrainingExecutor` plus
//!    `NoopLoraActivator`, then replay the terminal record with `get`;
//! 5. stop — `stop` cancels a queued run and gives its samples back to `pending`;
//! 6. deferred path — `PpoTrainingExecutor` fails with a typed `Unsupported`.
//!
//! Offline and deterministic: the store is in memory, the clock is the fixed epoch,
//! artifacts stay in a `TempDir`, and nothing prints an absolute path or a secret.

#![allow(
    // Every fallible call is documented at the SDK API it reaches; the helpers
    // below always succeed and are called for their narrative side effect.
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate
)]

mod common;

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use openjiuwen::agent_evolving::online::{
    FinishReason, LocalArtifactDescriptor, LocalTrainingExecutor, LoraActivator, MemoryOnlineStore,
    NoopLoraActivator, PpoTrainingExecutor, RewardMode, RunStatus, SampleLedger, SampleRecord,
    TaskRegistry, TaskSpec, TaskStatus, TrainingExecutor, TrainingRunRecord, TrainingRunner,
    TrainingRunnerConfig,
};
use serde_json::json;

use common::banner;

/// The one backend every component shares, the way a service shares one Redis.
type Store = Arc<MemoryOnlineStore>;

/// The RL task this example captures.
const TASK_ID: &str = "task-1";
/// The agent session the task belongs to.
const SESSION_ID: &str = "session-1";
/// The model identity every artifact must be named after (`model-a:v1`).
const MODEL_ID: &str = "model-a";
/// The policy the first batch rolls out on; the executor bumps it to `v1`.
const BASE_POLICY: &str = "base";
/// The sample owner the training runner batches for.
const USER_ID: &str = "user-1";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mode = common::setup()?;
    banner::title("26 online training service");
    banner::note(&format!(
        "run mode: {}; store in memory, clock at the fixed epoch, artifacts in a temp dir",
        mode.describe()
    ));

    let store: Store = Arc::new(MemoryOnlineStore::new());
    let artifacts = common::tempdir()?;
    let registry = TaskRegistry::new(Arc::clone(&store));
    let ledger = SampleLedger::new(Arc::clone(&store));

    let executor = Arc::new(LocalTrainingExecutor::new(artifacts.path()));
    let activator = Arc::new(NoopLoraActivator::new(Arc::clone(&store)));
    let runner = TrainingRunner::new(
        Arc::clone(&store),
        Arc::clone(&executor) as Arc<dyn TrainingExecutor>,
        Arc::clone(&activator) as Arc<dyn LoraActivator>,
        TrainingRunnerConfig {
            model_id: MODEL_ID.to_owned(),
            min_samples_for_training: 2,
            max_samples_per_run: 4,
            batch_size: 2,
            ..TrainingRunnerConfig::default()
        },
    );

    task_section(&registry).await?;
    capture_section(&registry).await?;
    ledger_section(&store, &ledger, artifacts.path()).await?;
    training_section(&ledger, &runner, &executor, &activator).await?;
    stop_section(&ledger, &runner).await?;
    deferred_section(&store, &ledger).await?;
    finish_section(&registry).await?;
    summary_section(&store, &ledger, &runner).await?;

    banner::step("done");
    Ok(())
}

/// Step 1: the task machine — `start` creates, a replay adopts, the session index
/// points at the task.
async fn task_section(registry: &TaskRegistry<Store>) -> Result<()> {
    banner::step("1 — task machine: start, replay, active-session lookup");

    let spec = TaskSpec {
        rl_task_id: TASK_ID.to_owned(),
        agent_session_id: SESSION_ID.to_owned(),
        model_id: MODEL_ID.to_owned(),
        policy_lora_name: BASE_POLICY.to_owned(),
        reward_mode: RewardMode::Terminal,
        policy_model: None,
    };
    let (record, created) = registry.start(&spec).await.context("start the RL task")?;
    ensure!(
        created,
        "the first start of a fresh id must create the task"
    );
    banner::note(&format!(
        "start `{}`: created = {created}, status `{}`, reward mode `{}`, policy `{}`",
        record.rl_task_id,
        record.status.as_str(),
        record.reward_mode.as_str(),
        record.policy_model
    ));

    // The same spec again is a replay: same record, `created = false`.
    let (replay, created_again) = registry.start(&spec).await?;
    ensure!(!created_again, "a replayed start adopts the record");
    banner::note(&format!(
        "replay start `{}`: created = {created_again}, status `{}`",
        replay.rl_task_id,
        replay.status.as_str()
    ));

    let active = registry.active_task_for_session(SESSION_ID).await?;
    banner::note(&format!(
        "active task for `{SESSION_ID}`: {}",
        active
            .as_ref()
            .map_or("none", |task| task.rl_task_id.as_str())
    ));
    Ok(())
}

/// Step 2: capture — reserve a capture id, commit stages its raw JSON, a second
/// capture is released without touching the first.
async fn capture_section(registry: &TaskRegistry<Store>) -> Result<()> {
    banner::step("2 — capture: reserve, commit stages raw JSON, discard releases");

    let sample = json!({
        "messages": [
            {"role": "user", "content": "which files are the largest?"},
            {"role": "assistant", "content": "run `du -ah . | sort -h | tail -n 20`"}
        ],
        "policy_version": BASE_POLICY,
    });

    let transition = registry
        .begin_capture(TASK_ID, "cap-1", None, "request-fingerprint-1")
        .await?;
    ensure!(
        transition.is_none(),
        "terminal mode never parks a turn transition"
    );
    banner::note("begin_capture `cap-1`: open, no turn transition in terminal mode");

    let committed = registry
        .commit_capture(TASK_ID, "cap-1", None, "response-fingerprint-1", &sample)
        .await?;
    ensure!(committed, "the first commit stages the sample");
    let staged = registry.pending_sample_ids(TASK_ID).await?;
    banner::note(&format!(
        "commit_capture `cap-1`: staged sample(s) [{}], {} capture(s) still open",
        join(&staged),
        registry.open_capture_count(TASK_ID).await?
    ));

    // The same response again is a no-op, which is what makes the pipeline replayable.
    let replayed = registry
        .commit_capture(TASK_ID, "cap-1", None, "response-fingerprint-1", &sample)
        .await?;
    ensure!(!replayed, "a replayed commit writes nothing");
    banner::note("replayed commit_capture `cap-1`: wrote = false");

    // A capture whose response never arrives is released, and only it.
    registry
        .begin_capture(TASK_ID, "cap-2", None, "request-fingerprint-2")
        .await?;
    let open_before = registry.open_capture_count(TASK_ID).await?;
    registry.discard_capture(TASK_ID, "cap-2", None).await?;
    let open_after = registry.open_capture_count(TASK_ID).await?;
    banner::note(&format!(
        "discard_capture `cap-2`: {open_before} open -> {open_after} open"
    ));

    banner::note(
        "capture only stages raw JSON under `rl:v1:task:task-1:pending:cap-1`; it never promotes \
         into the training ledger — `SampleLedger::save_once` does, and that is also the HTTP \
         upload path the service exposes",
    );
    Ok(())
}

/// Step 3: the ledger — first-wins publication, `stats()`, and a claim that is
/// refused because the batch is too small.
async fn ledger_section(
    store: &Store,
    ledger: &SampleLedger<Store>,
    artifacts: &Path,
) -> Result<()> {
    banner::step("3 — the sample ledger: first-wins publication and a batch that is too small");

    let samples = [
        sample("sample-1", 0.5),
        sample("sample-2", 0.75),
        sample("sample-3", 1.0),
    ];
    let saved = ledger
        .save_once(&samples)
        .await
        .context("save the first sample batch")?;
    banner::note(&format!(
        "save_once: saved {} sample(s): {}",
        saved.len(),
        join(&saved)
    ));

    // Replaying the upload saves nothing: the first writer of an id owns the record.
    let replayed = ledger.save_once(&samples).await?;
    banner::note(&format!(
        "replayed save_once: saved {} sample(s), every status stays `pending`",
        replayed.len()
    ));
    note_stats(&ledger.stats().await?, "ledger.stats()");

    // A runner needing ten samples refuses this queue with the service's envelope.
    let strict = TrainingRunner::new(
        Arc::clone(store),
        Arc::new(LocalTrainingExecutor::new(artifacts)),
        Arc::new(NoopLoraActivator::new(Arc::clone(store))),
        TrainingRunnerConfig {
            model_id: MODEL_ID.to_owned(),
            min_samples_for_training: 10,
            ..TrainingRunnerConfig::default()
        },
    );
    let refused = strict
        .claim()
        .await
        .err()
        .context("a claim must not start a run from three pending samples")?;
    banner::note(&format!(
        "claim with min_samples_for_training = 10: code `{}`, HTTP {} — {refused}",
        refused.code(),
        refused.http_status()
    ));
    Ok(())
}

/// Step 4: claim a batch, execute it through the local executor, activate the
/// artifact, and replay the terminal record.
async fn training_section(
    ledger: &SampleLedger<Store>,
    runner: &TrainingRunner<Store>,
    executor: &LocalTrainingExecutor,
    activator: &NoopLoraActivator<Store>,
) -> Result<()> {
    banner::step("4 — training: claim a fixed batch, run it, activate the artifact");

    let (queued, created, batch) = runner.claim().await.context("claim a training batch")?;
    ensure!(created, "the first claim must create the run");
    let batch_ids: Vec<String> = batch
        .iter()
        .map(|sample| sample.sample_id.clone())
        .collect();
    banner::note(&format!(
        "claim: {}, {} sample(s) [{}], rolled out on {}",
        describe_run(&queued),
        queued.sample_count,
        join(&batch_ids),
        counts(&queued.policy_versions)
    ));
    note_stats(&ledger.stats().await?, "ledger after the claim");

    let succeeded = runner
        .run(&queued.training_run_id)
        .await
        .context("execute the training run")?;
    ensure!(
        succeeded.status == RunStatus::Succeeded,
        "the local executor and the no-op activator succeed"
    );
    banner::note(&format!(
        "run: {}, policy `{}`",
        describe_run(&succeeded),
        succeeded.lora_name.as_deref().unwrap_or("none")
    ));

    // A terminal run is returned unchanged, so the caller can always re-read it.
    let replay = runner.get(&succeeded.training_run_id).await?;
    ensure!(replay == succeeded, "a terminal run is returned unchanged");
    banner::note("get: the terminal record is replayed unchanged (`run` is idempotent too)");

    let descriptor_path = executor.descriptor_path(None, &succeeded.training_run_id, 1);
    ensure!(
        descriptor_path.is_file(),
        "the local executor writes one descriptor per run"
    );
    let encoded = tokio::fs::read_to_string(&descriptor_path)
        .await
        .context("read the artifact descriptor")?;
    let descriptor: LocalArtifactDescriptor =
        serde_json::from_str(&encoded).context("decode the artifact descriptor")?;
    banner::note(&format!(
        "artifact `{}`: {} sample(s), mean reward {:.2}, best reward {:.2}, parent `{}`, written \
         as `{}/v{}.json`",
        descriptor.lora_name,
        descriptor.sample_count,
        descriptor.mean_reward,
        descriptor.best_reward,
        descriptor.base,
        succeeded.training_run_id,
        descriptor.version
    ));

    let activation = activator
        .recorded(&succeeded.training_run_id)
        .await?
        .context("the no-op activator records the switch")?;
    banner::note(&format!(
        "NoopLoraActivator: base model `{}`, policy `{}`, expected parent `{}`",
        activation.base_model, activation.lora_name, activation.expected_lora_name
    ));

    note_stats(&ledger.stats().await?, "ledger after the run");
    note_stats(&runner.stats().await?, "runner.stats()");
    Ok(())
}

/// Step 5: a second run is claimed and stopped while it is still queued, which
/// cancels it and returns its batch to `pending`.
async fn stop_section(ledger: &SampleLedger<Store>, runner: &TrainingRunner<Store>) -> Result<()> {
    banner::step("5 — the stop path: cancel a queued run and requeue its batch");

    let saved = ledger
        .save_once(&[sample("sample-4", 0.25), sample("sample-5", 0.9)])
        .await?;
    banner::note(&format!(
        "save_once: saved {} sample(s): {}",
        saved.len(),
        join(&saved)
    ));

    let (queued, created, _batch) = runner.claim().await?;
    ensure!(
        created,
        "two pending samples satisfy min_samples_for_training = 2"
    );
    banner::note(&format!("claim: {}", describe_run(&queued)));

    let stopped = runner.stop(&queued.training_run_id).await?;
    ensure!(
        stopped.status == RunStatus::Canceled,
        "a queued run is canceled without entering the executor"
    );
    banner::note(&format!(
        "stop: {}, {} sample(s) back to `pending`",
        describe_run(&stopped),
        stopped.sample_count
    ));

    let active = runner.active_run().await?;
    banner::note(&format!(
        "active_run: {}",
        active
            .as_ref()
            .map_or("none", |run| run.training_run_id.as_str())
    ));
    note_stats(&ledger.stats().await?, "ledger after the stop");
    Ok(())
}

/// Step 6: the GPU path is a typed `Unsupported`, both at the health probe and
/// when a run reaches the executor.
async fn deferred_section(store: &Store, ledger: &SampleLedger<Store>) -> Result<()> {
    banner::step("6 — the deferred PPO path: a typed Unsupported instead of fake training");

    let ppo: Arc<dyn TrainingExecutor> = Arc::new(PpoTrainingExecutor::new());
    let unhealth = ppo
        .check_health()
        .await
        .err()
        .context("the PPO executor must not claim health")?;
    banner::note(&format!(
        "PpoTrainingExecutor::check_health: kind `{}` — {unhealth}",
        unhealth.kind()
    ));

    let runner = TrainingRunner::new(
        Arc::clone(store),
        Arc::clone(&ppo),
        Arc::new(NoopLoraActivator::new(Arc::clone(store))),
        TrainingRunnerConfig {
            model_id: MODEL_ID.to_owned(),
            min_samples_for_training: 2,
            max_samples_per_run: 4,
            batch_size: 2,
            ..TrainingRunnerConfig::default()
        },
    );
    let (queued, created, _batch) = runner.claim().await?;
    ensure!(created, "the requeued batch is claimable again");
    let failed = runner.run(&queued.training_run_id).await?;
    ensure!(
        failed.status == RunStatus::Failed,
        "the deferred executor fails the run instead of pretending to train"
    );
    banner::note(&format!("run: {}", describe_run(&failed)));
    banner::note(&format!(
        "failure reason: {}",
        failed.failure_reason.as_deref().unwrap_or("none")
    ));
    note_stats(
        &ledger.stats().await?,
        "ledger after the failed run (the batch returned to `pending`)",
    );
    Ok(())
}

/// Step 7: close the capture window, then project the terminal reward onto the
/// staged samples.
async fn finish_section(registry: &TaskRegistry<Store>) -> Result<()> {
    banner::step("7 — closing the capture window: finish, then the terminal reward");

    let finished = registry.finish(TASK_ID, FinishReason::UserStopped).await?;
    ensure!(
        finished.status == TaskStatus::Finalized,
        "finish finalizes an active task"
    );
    banner::note(&format!(
        "finish: status `{}`, reason `{}`, stamp `{}`",
        finished.status.as_str(),
        finished
            .finish_reason
            .map_or("none", |reason| reason.as_str()),
        finished.finished_at.as_deref().unwrap_or("none")
    ));

    let covered = registry.record_reward(TASK_ID, 0.75).await?;
    let replayed = registry.record_reward(TASK_ID, 0.75).await?;
    ensure!(
        covered == replayed,
        "the terminal reward is projected exactly once"
    );
    banner::note(&format!(
        "record_reward 0.75: projected onto {covered} staged sample(s), a replay returns the same \
         count"
    ));
    banner::note(&format!(
        "staged samples after the projection: {} — the ledger's trained samples are untouched",
        registry.pending_sample_ids(TASK_ID).await?.len()
    ));
    Ok(())
}

/// Step 8: the final counters, the store's footprint, and what the run showed.
async fn summary_section(
    store: &Store,
    ledger: &SampleLedger<Store>,
    runner: &TrainingRunner<Store>,
) -> Result<()> {
    banner::step("8 — final snapshot");

    note_stats(&ledger.stats().await?, "sample ledger");
    note_stats(
        &runner.stats().await?,
        "runner.stats() over every run in the store",
    );
    banner::note(&format!(
        "in-memory store: {} keys, {} string values — the Redis backend writes the same `rl:v1` \
         layout under one prefix",
        store.keys().await.len(),
        store.snapshot().await.len()
    ));

    banner::note(
        "what this showed: one state machine — task -> capture -> ledger -> training run -> \
         activation — with every rule held in the store, so the in-memory backend stands in for \
         Redis and a restart resumes through `TrainingRunner::recover`",
    );
    banner::note(
        "what this showed: the local executor writes one `v1.json` descriptor per run, while the \
         GPU path (PPO over torch/verl/ray) is a typed `Unsupported`, never a silent no-op",
    );
    banner::note(
        "what this showed: capture stages samples but never publishes them — training begins only \
         from `SampleLedger::save_once`, the HTTP upload path — and the in-memory store forgets \
         everything at exit",
    );
    Ok(())
}

/// Builds one pending ledger sample of this task and session.
fn sample(sample_id: &str, reward: f64) -> SampleRecord {
    let payload = json!({ "policy_version": BASE_POLICY });
    SampleRecord {
        session_id: SESSION_ID.to_owned(),
        task_id: TASK_ID.to_owned(),
        ..SampleRecord::pending(sample_id, USER_ID, reward, payload)
    }
}

/// Renders a training run as its id, its status, and its stage.
fn describe_run(run: &TrainingRunRecord) -> String {
    format!(
        "`{}`: status `{}`, stage `{}`",
        run.training_run_id,
        run.status.as_str(),
        run.stage.as_str()
    )
}

/// Joins ids into one comma-separated list, or `none` for an empty batch.
fn join(ids: &[String]) -> String {
    if ids.is_empty() {
        return "none".to_owned();
    }
    ids.join(", ")
}

/// Renders a `name -> value` map as `name=value`, sorted by name.
fn counts<T: std::fmt::Display>(map: &BTreeMap<String, T>) -> String {
    if map.is_empty() {
        return "none".to_owned();
    }
    map.iter()
        .map(|(name, value)| format!("{name}={value}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Prints one `stats()` map (sample counters, run counters) as `name=count`.
fn note_stats(stats: &BTreeMap<String, usize>, label: &str) {
    banner::note(&format!("{label}: {}", counts(stats)));
}
