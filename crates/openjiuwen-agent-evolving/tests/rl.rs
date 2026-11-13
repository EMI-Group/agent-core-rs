//! Offline tests for the RL orchestration: packing, normalisation, policy and
//! rollout seams, the trainer loop, checkpoints, and the optimizer wiring.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;
use std::sync::Arc;

use openjiuwen_agent_evolving::rl::{
    token_estimate, BatchPacker, DeterministicPolicy, FakeRolloutEngine, OfflineRlOptimizer,
    OnlineRlOptimizer, Policy, RewardNormalisation, RewardNormaliser, RlBatch, RlCheckpoint,
    RlConfig, RlRolloutConfig, RlSample, RlStopReason, RlTask, RlTrainer, RlTrainerConfig,
    RlTrainingConfig, RolloutEngine, RolloutLog, RolloutWithReward, MESSAGE_TOKEN_OVERHEAD,
};
use openjiuwen_common::error::{Error, ErrorKind};
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio_util::sync::CancellationToken;

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-9,
        "expected {expected}, got {actual}"
    );
}

fn sample(id: &str, prompt: &str, response: &str, reward: f64) -> RlSample {
    RlSample::new(id, prompt, response, reward).expect("valid sample")
}

fn rollout(task: &str, turn: u64, reward: f64, group: Option<&str>) -> RolloutWithReward {
    RolloutWithReward {
        turn_id: turn,
        task_id: task.to_owned(),
        rollout_id: format!("{task}-rollout"),
        input_prompt_ids: vec![1, 2],
        output_response_ids: vec![3],
        reward,
        n_turns: 1,
        loss_mask: vec![1],
        group_id: group.map(str::to_owned),
    }
}

fn task(id: &str, prompt: &str, expected: &str, num_samples: u64) -> RlTask {
    RlTask {
        task_id: id.to_owned(),
        origin_task_id: None,
        task_sample: json!({
            "prompt": prompt,
            "expected": expected,
            "num_samples": num_samples,
        }),
        round_num: 0,
    }
}

fn scripted_policy(responses: &[(&str, &str)]) -> DeterministicPolicy {
    DeterministicPolicy::new(
        responses
            .iter()
            .map(|(prompt, response)| ((*prompt).to_owned(), (*response).to_owned()))
            .collect::<BTreeMap<_, _>>(),
    )
}

#[test]
fn token_estimate_is_deterministic_and_utf8_aware() {
    assert_eq!(MESSAGE_TOKEN_OVERHEAD, 4);
    assert_eq!(token_estimate(""), 0);
    assert_eq!(token_estimate("a"), 5);
    assert_eq!(token_estimate("abcd"), 5);
    assert_eq!(token_estimate("abcde"), 6);
    // Four Unicode scalar values, not eight UTF-8 bytes.
    assert_eq!(token_estimate("你好世界"), 5);
}

#[test]
fn samples_validate_identifier_prompt_and_reward() {
    assert!(RlSample::new("", "p", "r", 0.0).is_err());
    assert!(RlSample::new("id", "  ", "r", 0.0).is_err());
    assert!(RlSample::new("id", "p", "r", f64::NAN).is_err());
    assert!(RlSample::new("id", "p", "r", f64::INFINITY).is_err());
    let valid = sample("id", "ab", "cd", 0.5)
        .with_group_id("g")
        .with_metadata("source", json!("test"));
    assert_eq!(valid.group_id.as_deref(), Some("g"));
    assert_eq!(valid.metadata["source"], json!("test"));
    assert_eq!(
        valid.token_count(),
        token_estimate("ab") + token_estimate("cd")
    );
}

#[test]
fn packer_chunks_in_input_order_with_stable_ids() {
    let samples: Vec<RlSample> = (0..5)
        .map(|index| sample(&format!("s{index}"), "p", "r", f64::from(index)))
        .collect();
    let packer = BatchPacker::default().with_max_samples(2);
    let batches = packer.pack(&samples);
    assert_eq!(batches.len(), 3);
    assert_eq!(batches[0].batch_id, "batch-0");
    assert_eq!(batches[1].batch_id, "batch-1");
    assert_eq!(batches[2].batch_id, "batch-2");
    assert_eq!(batches[0].num_samples(), 2);
    assert_eq!(batches[1].num_samples(), 2);
    assert_eq!(batches[2].num_samples(), 1);
    let ids: Vec<&str> = batches
        .iter()
        .flat_map(|batch| batch.samples.iter().map(|item| item.sample_id.as_str()))
        .collect();
    assert_eq!(ids, ["s0", "s1", "s2", "s3", "s4"]);
    assert_eq!(batches[0].truncated, 0);
    assert_eq!(
        batches[0].total_tokens,
        batches[0]
            .samples
            .iter()
            .map(RlSample::token_count)
            .sum::<usize>()
    );
    assert_eq!(packer.pack(&[]).len(), 0);
    assert_close(batches[0].mean_reward(), 0.5);
    assert!(RlBatch::from_samples("manual", Vec::new()).is_empty());
}

#[test]
fn packer_truncates_oversized_samples_utf8_safely() {
    let long_response = "é".repeat(100);
    let item = sample("s0", "hello world", &long_response, 1.0);
    let packer = BatchPacker::default().with_max_tokens_per_sample(20);
    let batches = packer.pack(&[item]);
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].truncated, 1);
    let fitted = &batches[0].samples[0];
    assert!(fitted.token_count() <= 20);
    assert_eq!(fitted.prompt, "hello world");
    assert_eq!(fitted.response.chars().count(), 36);
    assert_eq!(batches[0].total_tokens, fitted.token_count());
}

#[test]
fn packer_splits_oversized_samples_when_truncation_is_disabled() {
    let samples = vec![
        sample("s0", "short", "ok", 0.0),
        sample("s1", "p", &"x".repeat(400), 1.0),
        sample("s2", "short", "ok", 0.5),
    ];
    let original_len = samples[1].response.len();
    let packer = BatchPacker::default()
        .with_max_tokens_per_sample(16)
        .with_truncate_long_samples(false);
    let batches = packer.pack(&samples);
    assert_eq!(batches.len(), 3);
    assert_eq!(batches[1].samples.len(), 1);
    assert_eq!(batches[1].samples[0].response.len(), original_len);
    assert_eq!(batches[1].truncated, 0);
    assert_eq!(batches[1].batch_id, "batch-1");
    assert_eq!(batches[2].samples[0].sample_id, "s2");
}

#[test]
fn group_centered_normalisation_uses_group_means() {
    let mut rollouts = vec![
        rollout("t", 0, 1.0, Some("a")),
        rollout("t", 1, 3.0, Some("a")),
        rollout("t", 2, 2.0, Some("b")),
        rollout("t", 3, 9.0, None),
    ];
    let stats = RewardNormaliser::new(RewardNormalisation::GroupCentered).normalise(&mut rollouts);
    assert_close(rollouts[0].reward, -1.0);
    assert_close(rollouts[1].reward, 1.0);
    assert_close(rollouts[2].reward, 0.0);
    assert_close(rollouts[3].reward, 0.0);
    assert_eq!(stats.count, 4);
    assert_eq!(stats.groups.len(), 3);
    assert_close(stats.groups["a"].mean, 0.0);
    assert_close(stats.groups["a"].min, -1.0);
    assert_close(stats.groups["a"].max, 1.0);
    let expected_mean = (-1.0 + 1.0 + 0.0 + 0.0) / 4.0;
    assert!((stats.mean - expected_mean).abs() < 1e-12);
}

#[test]
fn z_score_and_min_max_have_zero_variance_guards() {
    let mut grouped = vec![
        rollout("t", 0, 1.0, Some("a")),
        rollout("t", 1, 3.0, Some("a")),
        rollout("t", 2, 7.0, Some("b")),
        rollout("t", 3, 7.0, Some("b")),
    ];
    let stats = RewardNormaliser::new(RewardNormalisation::GroupZScore).normalise(&mut grouped);
    assert!((grouped[0].reward + 1.0).abs() < 1e-12);
    assert!((grouped[1].reward - 1.0).abs() < 1e-12);
    assert_close(grouped[2].reward, 0.0);
    assert_close(grouped[3].reward, 0.0);
    assert_close(stats.groups["b"].max, 0.0);

    let mut spread = vec![
        rollout("t", 0, 1.0, Some("a")),
        rollout("t", 1, 3.0, Some("a")),
        rollout("t", 2, 5.0, Some("b")),
        rollout("t", 3, 5.0, Some("b")),
    ];
    let stats = RewardNormaliser::new(RewardNormalisation::MinMax).normalise(&mut spread);
    assert_close(spread[0].reward, 0.0);
    assert_close(spread[1].reward, 1.0);
    assert_close(spread[2].reward, 0.0);
    assert_close(spread[3].reward, 0.0);
    assert_close(stats.groups["a"].mean, 0.5);
}

#[test]
fn identity_mode_and_clipping_apply_after_normalisation() {
    let mut unchanged = vec![
        rollout("t", 0, -4.0, Some("a")),
        rollout("t", 1, 8.0, Some("a")),
    ];
    RewardNormaliser::new(RewardNormalisation::None).normalise(&mut unchanged);
    assert_close(unchanged[0].reward, -4.0);
    assert_close(unchanged[1].reward, 8.0);

    let mut clipped = vec![
        rollout("t", 0, -4.0, Some("a")),
        rollout("t", 1, 8.0, Some("a")),
    ];
    let stats = RewardNormaliser::new(RewardNormalisation::GroupCentered)
        .with_clip(1.0, -1.0)
        .normalise(&mut clipped);
    assert_close(clipped[0].reward, -1.0);
    assert_close(clipped[1].reward, 1.0);
    assert_close(stats.min, -1.0);
    assert_close(stats.max, 1.0);
}

#[test]
fn normalisation_sanitises_non_finite_rewards() {
    let mut rollouts = vec![
        rollout("t", 0, f64::NAN, Some("a")),
        rollout("t", 1, 1.0, Some("a")),
    ];
    let stats = RewardNormaliser::new(RewardNormalisation::GroupCentered).normalise(&mut rollouts);
    assert!(rollouts.iter().all(|item| item.reward.is_finite()));
    assert!(stats.mean.is_finite());
    assert!(stats.std_dev.is_finite());
    assert_close(rollouts[0].reward, -0.5);
    assert_close(rollouts[1].reward, 0.5);
}

#[test]
fn rollout_log_projects_samples_and_statistics() {
    let mut log = RolloutLog::new("t");
    let mut first = rollout("t", 0, 0.5, Some("g"));
    first.rollout_id = "r".to_owned();
    first.input_prompt_ids = vec![1, 2];
    first.output_response_ids = vec![3];
    log.push(first);
    let mut second = rollout("t", 1, f64::NAN, None);
    second.rollout_id = "r".to_owned();
    second.input_prompt_ids.clear();
    second.output_response_ids.clear();
    log.push(second);

    assert_eq!(log.len(), 2);
    assert!(!log.is_empty());
    assert!((log.mean_reward() - 0.25).abs() < 1e-12);

    let samples = log.to_samples();
    assert_eq!(samples.len(), 2);
    assert_eq!(samples[0].sample_id, "r-0-0");
    assert_eq!(samples[0].prompt, "1 2");
    assert_eq!(samples[0].response, "3");
    assert_close(samples[0].reward, 0.5);
    assert_eq!(samples[0].group_id.as_deref(), Some("g"));
    assert_eq!(samples[0].metadata["turn_id"], json!(0));
    assert_eq!(samples[1].prompt, "t");
    assert_close(samples[1].reward, 0.0);
    assert_eq!(samples[1].group_id, None);
}

#[tokio::test]
async fn deterministic_policy_acts_updates_and_round_trips_state() {
    let policy = scripted_policy(&[("hi", "hello")]).with_fallback_prefix("p");
    let cancel = CancellationToken::new();
    assert_eq!(policy.name(), "deterministic");
    assert_eq!(policy.act("hi", &cancel).await.expect("act"), "hello");
    let first = policy.act("unknown", &cancel).await.expect("act");
    let second = policy.act("unknown", &cancel).await.expect("act");
    assert_eq!(first, second);
    assert!(first.starts_with("p:"));

    let batch = RlBatch::from_samples(
        "batch-0",
        vec![sample("s0", "a", "b", 0.5), sample("s1", "a", "c", 1.0)],
    );
    let update = policy.update(&batch).await.expect("update");
    assert_eq!(update.samples, 2);
    assert!((update.mean_reward - 0.75).abs() < 1e-12);
    assert_close(update.best_reward, 1.0);
    assert_eq!(update.parameters["updates"], json!(1));

    let second_update = policy.update(&batch).await.expect("update");
    assert_eq!(second_update.parameters["updates"], json!(2));

    let state = policy.parameters();
    let restored = scripted_policy(&[]);
    assert_eq!(restored.parameters(), Value::Null);
    Policy::restore(&restored, &state).expect("restore");
    assert_eq!(restored.parameters(), state);
}

#[tokio::test]
async fn policy_act_honours_cancellation() {
    let policy = scripted_policy(&[]);
    let cancel = CancellationToken::new();
    cancel.cancel();
    let error = policy.act("prompt", &cancel).await.expect_err("cancelled");
    assert!(matches!(error, Error::Cancelled));
}

#[tokio::test]
async fn fake_engine_rolls_out_with_exact_match_rewards() {
    let engine = FakeRolloutEngine::new();
    let policy: Arc<dyn Policy> = Arc::new(scripted_policy(&[("q", "a")]));
    let cancel = CancellationToken::new();
    let rollouts = engine
        .roll(&task("t", "q", "a", 2), Arc::clone(&policy), &cancel)
        .await
        .expect("roll");
    assert_eq!(engine.name(), "fake");
    assert_eq!(rollouts.len(), 2);
    assert!(rollouts
        .iter()
        .all(|item| (item.reward - 1.0).abs() < 1e-12));
    assert_eq!(rollouts[0].task_id, "t");
    assert_eq!(rollouts[0].rollout_id, "t-0");
    assert_eq!(rollouts[1].turn_id, 1);
    assert_eq!(rollouts[0].group_id.as_deref(), Some("q"));
    assert_eq!(rollouts[0].input_prompt_ids, vec![u32::from('q')]);

    let wrong: Arc<dyn Policy> = Arc::new(scripted_policy(&[("q", "b")]));
    let rollouts = engine
        .roll(&task("t", "q", "a", 1), wrong, &cancel)
        .await
        .expect("roll");
    assert_close(rollouts[0].reward, 0.0);
}

#[tokio::test]
async fn fake_engine_accepts_custom_scoring_and_rejects_cancellation() {
    let engine = FakeRolloutEngine::new()
        .with_reward_fn(
            |_expected, response| {
                if response.starts_with('x') {
                    0.5
                } else {
                    0.0
                }
            },
        )
        .with_sample_count(1);
    let policy: Arc<dyn Policy> = Arc::new(scripted_policy(&[("q", "xyz")]));
    let cancel = CancellationToken::new();
    let rollouts = engine
        .roll(&task("t", "q", "a", 9), policy, &cancel)
        .await
        .expect("roll");
    assert_eq!(rollouts.len(), 1);
    assert_close(rollouts[0].reward, 0.5);

    let cancelled = CancellationToken::new();
    cancelled.cancel();
    let error = engine
        .roll(
            &task("t", "q", "a", 1),
            Arc::new(scripted_policy(&[])) as Arc<dyn Policy>,
            &cancelled,
        )
        .await
        .expect_err("cancelled");
    assert!(matches!(error, Error::Cancelled));
}

#[tokio::test]
async fn trainer_runs_configured_epochs_and_reports() {
    let config = RlTrainerConfig {
        run_id: Some("run-1".to_owned()),
        epochs: 3,
        early_stop_reward: 2.0,
        ..RlTrainerConfig::default()
    };
    let trainer = RlTrainer::new(config)
        .with_normaliser(RewardNormaliser::new(RewardNormalisation::None))
        .with_packer(BatchPacker::default().with_max_samples(8));
    let cancel = CancellationToken::new();
    let policy: Arc<dyn Policy> = Arc::new(scripted_policy(&[("q", "a")]));
    let engine = FakeRolloutEngine::new();
    let report = trainer
        .train(policy, &engine, &[task("t", "q", "a", 2)], &cancel)
        .await
        .expect("train");

    assert_eq!(report.run_id, "run-1");
    assert_eq!(report.epochs_run, 3);
    assert_eq!(report.stopped_reason, RlStopReason::Completed);
    assert_eq!(report.epochs.len(), 3);
    assert_eq!(
        report
            .epochs
            .iter()
            .map(|epoch| epoch.epoch)
            .collect::<Vec<_>>(),
        [0, 1, 2]
    );
    assert!(report.epochs.iter().all(|epoch| epoch.updated));
    assert!(report.epochs.iter().all(|epoch| epoch.samples == 2));
    assert!((report.epochs[0].mean_reward - 1.0).abs() < 1e-12);
    assert_close(report.best_reward, 1.0);
    let best = report.best_batch.expect("best batch");
    assert_eq!(best.num_samples(), 2);
    assert!(report.epochs.iter().all(|epoch| epoch.checkpoint.is_none()));
}

#[tokio::test]
async fn trainer_early_stops_at_threshold() {
    let config = RlTrainerConfig {
        epochs: 5,
        early_stop_reward: 0.5,
        ..RlTrainerConfig::default()
    };
    let trainer =
        RlTrainer::new(config).with_normaliser(RewardNormaliser::new(RewardNormalisation::None));
    let cancel = CancellationToken::new();
    let report = trainer
        .train(
            Arc::new(scripted_policy(&[("q", "a")])),
            &FakeRolloutEngine::new(),
            &[task("t", "q", "a", 1)],
            &cancel,
        )
        .await
        .expect("train");
    assert_eq!(report.stopped_reason, RlStopReason::EarlyStop);
    assert_eq!(report.epochs_run, 1);
}

#[tokio::test]
async fn trainer_reports_cancellation_and_empty_tasks() {
    let trainer = RlTrainer::new(RlTrainerConfig::default());
    let engine = FakeRolloutEngine::new();
    let policy: Arc<dyn Policy> = Arc::new(scripted_policy(&[("q", "a")]));
    let cancel = CancellationToken::new();
    cancel.cancel();
    let report = trainer
        .train(policy, &engine, &[task("t", "q", "a", 1)], &cancel)
        .await
        .expect("train");
    assert_eq!(report.stopped_reason, RlStopReason::Cancelled);
    assert_eq!(report.epochs_run, 0);
    assert_eq!(report.epochs.len(), 0);

    let report = trainer
        .train(
            Arc::new(scripted_policy(&[])),
            &engine,
            &[],
            &CancellationToken::new(),
        )
        .await
        .expect("train");
    assert_eq!(report.stopped_reason, RlStopReason::NoTasks);
    assert_eq!(report.run_id, "rl-run");
    assert!(report.best_batch.is_none());
}

#[tokio::test]
async fn checkpoints_round_trip_and_latest_is_newest() {
    let dir = TempDir::new().expect("temp dir");
    let metrics = BTreeMap::from([("mean_reward".to_owned(), 0.5)]);
    let first = RlCheckpoint::new("run", 1, 0.25, json!({"w": 1}), metrics.clone());
    let path = first.save_to_dir(dir.path()).await.expect("save");
    assert_eq!(
        path.file_name().and_then(|name| name.to_str()),
        Some("epoch-1.json")
    );
    let loaded = RlCheckpoint::load_from_path(&path).await.expect("load");
    assert_eq!(loaded, first);
    let missing = TempDir::new().expect("temp dir");
    assert_eq!(
        RlCheckpoint::latest_in(missing.path())
            .await
            .expect("latest"),
        None
    );

    let third = RlCheckpoint::new("run", 3, 0.75, json!({"w": 3}), metrics);
    third.save_to_dir(dir.path()).await.expect("save");
    let latest = RlCheckpoint::latest_in(dir.path())
        .await
        .expect("latest")
        .expect("checkpoint");
    assert_eq!(latest.epoch, 3);
    assert_close(latest.best_reward, 0.75);
    assert!(dir.path().join("latest.json").exists());

    let invalid = RlCheckpoint {
        version: "v2".to_owned(),
        ..first.clone()
    };
    assert_eq!(
        invalid.validate().expect_err("bad version").kind(),
        ErrorKind::Validation
    );
    let non_finite = RlCheckpoint::new("run", 0, f64::NAN, Value::Null, BTreeMap::new());
    assert!(non_finite.save_to_dir(dir.path()).await.is_err());
}

#[tokio::test]
async fn offline_optimizer_train_with_uses_config_and_checkpoints() {
    let dir = TempDir::new().expect("temp dir");
    let mut optimizer = OfflineRlOptimizer::new(RlConfig {
        training: RlTrainingConfig {
            epoch_num: 2,
            save_path: Some(dir.path().to_string_lossy().into_owned()),
            save_freq: 1,
            ..RlTrainingConfig::default()
        },
        rollout: RlRolloutConfig {
            rollout_n: 1,
            ..RlRolloutConfig::default()
        },
        ..RlConfig::default()
    });
    assert!(optimizer.register_reward("exact", json!({"kind": "exact_match"})));
    let report = optimizer
        .train_with(
            Arc::new(scripted_policy(&[("q", "a")])),
            &FakeRolloutEngine::new(),
            &[task("t", "q", "a", 1)],
            &CancellationToken::new(),
        )
        .await
        .expect("train_with");
    assert_eq!(report.epochs_run, 2);
    assert!(report.epochs.iter().all(|epoch| epoch.checkpoint.is_some()));
    let latest = RlCheckpoint::latest_in(dir.path())
        .await
        .expect("latest")
        .expect("checkpoint");
    assert_eq!(latest.epoch, 1);
    assert_eq!(latest.run_id, "rl-t");
}

#[tokio::test]
async fn deferred_entry_points_keep_their_unsupported_contract() {
    let mut offline = OfflineRlOptimizer::new(RlConfig::default());
    let error = offline.train().await.expect_err("deferred");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.to_string().contains("OfflineRlOptimizer::train"));
    assert!(offline
        .start_training()
        .await
        .expect_err("deferred")
        .to_string()
        .contains("PPO/verl/ray"));

    let mut online = OnlineRlOptimizer::new(RlConfig::default());
    for (error, expected) in [
        (
            online.start_training().await.expect_err("deferred"),
            "OnlineRlOptimizer::start_training",
        ),
        (
            online
                .train_on_batch(json!({"batch": 1}))
                .await
                .expect_err("deferred"),
            "OnlineRlOptimizer::train_on_batch",
        ),
        (
            online.export_lora().expect_err("deferred"),
            "OnlineRlOptimizer::export_lora",
        ),
    ] {
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error.to_string().contains(expected));
        assert!(error.to_string().contains("Redis/vLLM/verl"));
    }
}
