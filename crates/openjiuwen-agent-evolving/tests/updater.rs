//! Offline tests for the parameter-update pipeline: the update contract, the
//! apply helpers, single- and multi-dimension updaters, and the RL update
//! pipeline over the `rl` seams.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use async_trait::async_trait;
use openjiuwen_agent_evolving::dataset::{Case, EvaluatedCase};
use openjiuwen_agent_evolving::rl::{
    BatchPacker, Policy, RlBatch, RlCheckpoint, RlSample, RlUpdate, RolloutWithReward,
    CHECKPOINT_VERSION,
};
use openjiuwen_agent_evolving::signal::EvolutionTarget;
use openjiuwen_agent_evolving::trajectory::{Trajectory, TrajectoryOutcome};
use openjiuwen_agent_evolving::updater::{
    compute_advantages, execute_updates, normalize_update_value, signals_from_cases,
    summarize_apply_results, AdvantageReport, GradientOptimizer, GpuGradientOptimizer,
    MultiDimUpdater, OfflinePolicyOptimizer, RecordingOptimizer, RecordingTarget, RlUpdatePipeline,
    SingleDimUpdater, TextualOptimizer, Update, UpdateConfig, UpdateDomain, UpdateMode,
    UpdateTarget, UpdateValue, Updater, Updates, EXPERIENCES_TARGET, POLICY_TARGET,
};
use openjiuwen_common::error::{Error, ErrorKind};
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio_util::sync::CancellationToken;

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

fn trajectory(id: &str) -> Trajectory {
    let mut trajectory = Trajectory::new(id).expect("trajectory");
    trajectory.outcome = TrajectoryOutcome::Success;
    trajectory
}

fn signal(target: EvolutionTarget) -> openjiuwen_agent_evolving::signal::EvolutionSignal {
    openjiuwen_agent_evolving::signal::EvolutionSignal::new("evaluated", target, "score=0.00")
        .expect("signal")
}

fn evaluated(score: f64, reason: &str) -> EvaluatedCase {
    let case = Case::new(
        [("question".to_owned(), json!("2+2?"))]
            .into_iter()
            .collect(),
        [("answer".to_owned(), json!("4"))].into_iter().collect(),
    )
    .expect("case")
    .with_case_id("case-1");
    EvaluatedCase::new(case, None, score, reason).expect("evaluated case")
}

fn sample(id: &str, reward: f64, group: Option<&str>) -> RlSample {
    let sample = RlSample::new(id, "prompt", "response", reward).expect("sample");
    match group {
        Some(group) => sample.with_group_id(group),
        None => sample,
    }
}

/// A policy that records the batches it is updated with.
#[derive(Debug, Default)]
struct RecordingPolicy {
    batches: Mutex<Vec<RlBatch>>,
    parameters: Mutex<Value>,
}

impl RecordingPolicy {
    fn batches(&self) -> Vec<RlBatch> {
        lock(&self.batches).clone()
    }
}

#[async_trait]
impl Policy for RecordingPolicy {
    fn name(&self) -> &'static str {
        "recording"
    }

    async fn act(&self, prompt: &str, _cancel: &CancellationToken) -> Result<String, Error> {
        Ok(format!("echo:{prompt}"))
    }

    async fn update(&self, batch: &RlBatch) -> Result<RlUpdate, Error> {
        let parameters = {
            let mut batches = lock(&self.batches);
            batches.push(batch.clone());
            json!({"updates": batches.len(), "last_mean": batch.mean_reward()})
        };
        *lock(&self.parameters) = parameters.clone();
        Ok(RlUpdate {
            samples: batch.samples.len(),
            mean_reward: batch.mean_reward(),
            best_reward: batch.mean_reward(),
            parameters,
        })
    }

    fn parameters(&self) -> Value {
        lock(&self.parameters).clone()
    }
}

/// A gradient optimizer that answers with a scripted update map.
struct ScriptedOptimizer {
    name: &'static str,
    updates: Updates,
    gpu: bool,
}

#[async_trait]
impl GradientOptimizer for ScriptedOptimizer {
    fn name(&self) -> &'static str {
        self.name
    }

    async fn gradients(
        &self,
        _batch: &RlBatch,
        _advantages: &AdvantageReport,
    ) -> Result<Updates, Error> {
        Ok(self.updates.clone())
    }

    fn requires_gpu(&self) -> bool {
        self.gpu
    }
}

fn rollout(reward: f64, group: Option<&str>) -> RolloutWithReward {
    RolloutWithReward {
        turn_id: 0,
        task_id: "task".to_owned(),
        rollout_id: "rollout".to_owned(),
        input_prompt_ids: vec![7],
        output_response_ids: vec![8],
        reward,
        n_turns: 1,
        loss_mask: vec![1],
        group_id: group.map(str::to_owned),
    }
}

// ---------------------------------------------------------------------------
// Update contract and apply helpers
// ---------------------------------------------------------------------------

#[test]
fn normalization_mirrors_the_python_compatibility_rules() {
    let raw = normalize_update_value(&Update::raw(json!("text")), "body");
    assert_eq!(raw.mode, UpdateMode::Replace);
    assert_eq!(raw.payload, json!("text"));

    let experiences = normalize_update_value(&Update::raw(json!(["e"])), EXPERIENCES_TARGET);
    assert_eq!(experiences.mode, UpdateMode::Append);
    assert_eq!(
        experiences.change_type.as_deref(),
        Some("skill_experience_entry")
    );
    assert_eq!(
        experiences.metadata.get("change_type"),
        Some(&json!("skill_experience_entry"))
    );

    let structured = Update::Structured(UpdateValue::new(json!(1)));
    assert_eq!(
        normalize_update_value(&structured, EXPERIENCES_TARGET).mode,
        UpdateMode::Replace
    );
}

#[test]
fn execute_updates_applies_in_write_order_and_reports_none_last() {
    let target = Arc::new(RecordingTarget::new());
    let mut operators = BTreeMap::new();
    let erased: Arc<dyn UpdateTarget> = Arc::clone(&target);
    operators.insert("op-1".to_owned(), erased);

    let mut updates = Updates::new();
    updates.insert(("op-1".to_owned(), "body".to_owned()), None);
    updates.insert_update("op-1", "description", json!("second-written"));
    updates.insert_update("op-2", "body", json!("missing operator"));

    let results = execute_updates(&operators, &updates);
    assert_eq!(results.len(), 3);
    assert_eq!(
        results
            .iter()
            .map(|result| (result.operator_id.as_str(), result.target.as_str()))
            .collect::<Vec<_>>(),
        [("op-1", "description"), ("op-2", "body"), ("op-1", "body")]
    );
    assert!(results[0].ok());
    assert_eq!(results[1].errors, ["operator not found: op-2"]);
    assert_eq!(results[2].errors, ["update value is None"]);
    let summary = summarize_apply_results(&results);
    assert_eq!((summary.total, summary.applied, summary.failed), (3, 1, 2));
    assert_eq!(target.written_targets(), ["description"]);
}

// ---------------------------------------------------------------------------
// Single-dimension updater
// ---------------------------------------------------------------------------

#[tokio::test]
async fn single_dim_updater_feeds_trajectories_then_applies_its_updates() {
    let target = Arc::new(RecordingTarget::new());
    let mut operators = BTreeMap::new();
    operators.insert("agent/llm_1".to_owned(), Arc::clone(&target));
    let mut updater = SingleDimUpdater::new(RecordingOptimizer::new());
    assert_eq!(updater.bind(&operators, None), 1);

    let trajectories = [trajectory("t-1"), trajectory("t-2")];
    let signals = [signal(EvolutionTarget::Description)];
    let updates = updater
        .process(&trajectories, &signals, &UpdateConfig::new())
        .await
        .expect("process");

    assert_eq!(updater.optimizer().trajectory_ids(), ["t-1", "t-2"]);
    assert_eq!(updater.optimizer().backward_calls(), 1);
    assert_eq!(updater.optimizer().steps(), 1);

    let results = execute_updates(&operators, &updates);
    assert!(summarize_apply_results(&results).is_ok());
    assert_eq!(
        target.value("agent/llm_1", "description"),
        Some(json!({"trajectories": 2, "signals": 1, "step": 0}))
    );
}

#[tokio::test]
async fn single_dim_update_filters_cases_by_the_threshold_inclusively() {
    let cases = [evaluated(0.0, "wrong"), evaluated(0.5, "half"), evaluated(0.9, "right")];
    let mut updater = SingleDimUpdater::new(RecordingOptimizer::new());
    let mut operators = BTreeMap::new();
    operators.insert("agent/llm_1".to_owned(), Arc::new(RecordingTarget::new()));
    updater.bind(&operators, None);

    updater
        .update(
            &[trajectory("t-1")],
            &cases,
            &UpdateConfig::new().with_score_threshold(0.5),
        )
        .await
        .expect("update");
    let seen = &updater.optimizer().signals()[0];
    assert_eq!(seen.len(), 1, "0.5 is skipped by the inclusive threshold");
    assert_eq!(seen[0].signal_type(), "low_score");
    assert_eq!(seen[0].context.get("score"), Some(&json!(0.0)));

    updater
        .update(&[trajectory("t-2")], &cases, &UpdateConfig::new())
        .await
        .expect("update without a threshold");
    let seen = &updater.optimizer().signals()[1];
    assert_eq!(seen.len(), 3);
    assert_eq!(
        seen.iter()
            .map(|signal| signal.signal_type())
            .collect::<Vec<_>>(),
        ["low_score", "evaluated", "evaluated"]
    );
}

#[test]
fn signals_from_cases_rejects_a_non_finite_threshold() {
    let error = signals_from_cases(&[evaluated(0.1, "x")], Some(f64::INFINITY)).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(signals_from_cases(&[evaluated(0.1, "x")], Some(0.5)).is_ok());
}

#[tokio::test]
async fn single_dim_state_round_trips_through_a_fresh_updater() {
    let mut operators = BTreeMap::new();
    operators.insert("agent/llm_1".to_owned(), Arc::new(RecordingTarget::new()));
    let mut updater = SingleDimUpdater::new(
        RecordingOptimizer::new().with_requires_forward_data(true),
    );
    assert!(updater.requires_forward_data());
    updater.bind(&operators, Some(&["description".to_owned()]));
    updater
        .process(&[trajectory("t-1")], &[signal(EvolutionTarget::Description)], &UpdateConfig::new())
        .await
        .expect("process");
    let state = updater.get_state();

    let mut restored = SingleDimUpdater::new(RecordingOptimizer::new());
    restored.load_state(&state).expect("load state");
    assert_eq!(restored.optimizer().bound(), ["agent/llm_1"]);
    assert_eq!(restored.optimizer().bound_targets(), Some(["description".to_owned()].as_slice()));
    assert_eq!(restored.optimizer().steps(), 1);
    assert_eq!(restored.optimizer().trajectory_ids(), Vec::<String>::new());
    assert!(restored.load_state(&json!("nope")).is_err());
    assert!(restored.load_state(&json!({"wrong": 1})).is_err());
}

#[tokio::test]
async fn a_boxed_optimizer_drives_the_single_dim_updater() {
    let mut operators = BTreeMap::new();
    operators.insert("agent/llm_1".to_owned(), Arc::new(RecordingTarget::new()));
    let boxed: Box<dyn TextualOptimizer> = Box::new(RecordingOptimizer::new());
    let mut updater = SingleDimUpdater::new(boxed);
    assert_eq!(updater.bind(&operators, None), 1);
    let updates = updater
        .process(&[trajectory("t-1")], &[signal(EvolutionTarget::Body)], &UpdateConfig::new())
        .await
        .expect("process");
    assert!(updates.contains_key("agent/llm_1", "description"));
}

// ---------------------------------------------------------------------------
// Multi-dimension updater
// ---------------------------------------------------------------------------

fn domain_operators() -> (BTreeMap<String, Arc<dyn UpdateTarget>>, Arc<RecordingTarget>, Arc<RecordingTarget>) {
    let llm = Arc::new(RecordingTarget::new());
    let tool = Arc::new(RecordingTarget::with_domain(UpdateDomain::Tool));
    let mut operators = BTreeMap::new();
    let llm_target: Arc<dyn UpdateTarget> = Arc::clone(&llm);
    let tool_target: Arc<dyn UpdateTarget> = Arc::clone(&tool);
    operators.insert("agent/llm_1".to_owned(), llm_target);
    operators.insert("agent/tool_1".to_owned(), tool_target);
    (operators, llm, tool)
}

#[tokio::test]
async fn multi_dim_applies_domains_in_the_documented_order() {
    let (operators, _, _) = domain_operators();
    let mut updater = MultiDimUpdater::new()
        .with_optimizer(UpdateDomain::Tool, Box::new(RecordingOptimizer::new()))
        .with_optimizer(UpdateDomain::Llm, Box::new(RecordingOptimizer::new()))
        .with_optimizer(UpdateDomain::Memory, Box::new(RecordingOptimizer::new()));
    assert_eq!(
        updater.domains(),
        [UpdateDomain::Llm, UpdateDomain::Tool, UpdateDomain::Memory]
    );
    assert_eq!(updater.bind(&operators, None), 2);

    let updates = updater
        .process(
            &[trajectory("t-1")],
            &[signal(EvolutionTarget::Tool), signal(EvolutionTarget::Instruction)],
            &UpdateConfig::new(),
        )
        .await
        .expect("process");
    let keys: Vec<(&str, &str)> = updates
        .iter()
        .map(|(operator_id, target, _)| (operator_id, target))
        .collect();
    assert_eq!(keys, [("agent/llm_1", "description"), ("agent/tool_1", "description")]);
    // The memory domain owns no operator and received no signal, so it was never
    // fed or stepped; each stepped domain saw the whole trajectory batch once.
    let memory = updater.optimizer(UpdateDomain::Memory).expect("memory");
    assert_eq!(memory.state()["bound"], json!([]));
    assert_eq!(memory.state()["steps"], json!(0));
    assert_eq!(updater.get_state()["domains"]["llm"]["steps"], json!(1));
    assert_eq!(updater.get_state()["domains"]["tool"]["trajectories"], json!(1));
}

#[tokio::test]
async fn multi_dim_backward_before_bind_is_a_validation_error() {
    let mut updater = MultiDimUpdater::new()
        .with_optimizer(
            UpdateDomain::Llm,
            Box::new(RecordingOptimizer::new().with_requires_forward_data(true)),
        )
        .with_optimizer(UpdateDomain::Tool, Box::new(RecordingOptimizer::new()));
    assert!(updater.requires_forward_data());
    let error = updater
        .process(&[], &[signal(EvolutionTarget::Skill)], &UpdateConfig::new())
        .await
        .expect_err("unbound domain optimizer");
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[tokio::test]
async fn multi_dim_rejects_results_that_collide_across_domains() {
    let (operators, _, _) = domain_operators();
    let mut llm = Updates::new();
    llm.insert_update("agent/llm_1", "description", json!("llm"));
    let mut tool = Updates::new();
    tool.insert_update("agent/llm_1", "description", json!("tool"));
    let mut updater = MultiDimUpdater::new()
        .with_optimizer(
            UpdateDomain::Llm,
            Box::new(RecordingOptimizer::new().with_updates(vec![llm])),
        )
        .with_optimizer(
            UpdateDomain::Tool,
            Box::new(RecordingOptimizer::new().with_updates(vec![tool])),
        );
    updater.bind(&operators, None);
    let error = updater
        .process(
            &[],
            &[signal(EvolutionTarget::Body), signal(EvolutionTarget::Tool)],
            &UpdateConfig::new(),
        )
        .await
        .expect_err("colliding keys");
    assert!(error.to_string().contains("duplicate update"));
}

#[test]
fn multi_dim_state_rejects_unknown_domains() {
    let mut updater = MultiDimUpdater::new()
        .with_optimizer(UpdateDomain::Llm, Box::new(RecordingOptimizer::new()));
    assert!(updater
        .load_state(&json!({"domains": {"audio": {}}}))
        .is_err());
    assert!(updater.load_state(&json!([])).is_err());
    let state = updater.get_state();
    assert!(updater.load_state(&state).is_ok());
    assert!(updater
        .load_state(&json!({"domains": {"tool": {}}}))
        .is_err());
}

// ---------------------------------------------------------------------------
// RL advantages and the update pipeline
// ---------------------------------------------------------------------------

#[test]
fn advantages_are_group_centred_and_report_per_group_stats() {
    let mut rollouts = vec![
        rollout(1.0, Some("g1")),
        rollout(0.0, Some("g1")),
        rollout(2.0, Some("g2")),
        rollout(4.0, None),
    ];
    let report = compute_advantages(&mut rollouts);
    assert_eq!(report.advantages, [0.5, -0.5, 0.0, 0.0]);
    assert_eq!(report.rewards, [1.0, 0.0, 2.0, 4.0]);
    assert_eq!(
        rollouts.iter().map(|rollout| rollout.reward).collect::<Vec<_>>(),
        [0.5, -0.5, 0.0, 0.0]
    );
    let group = report.group("g1").expect("g1");
    assert_eq!(group.count, 2);
    assert_eq!(group.mean_reward, 0.5);
    assert_eq!(group.min_advantage, -0.5);
    assert_eq!(report.group("g2").expect("g2").mean_advantage, 0.0);
    assert_eq!(report.fallback_groups, [String::new()]);
    assert!(report.group("").expect("fallback").fallback);
    assert_eq!(report.len(), 4);
    assert_eq!(report.best_reward(), 4.0);
}

#[tokio::test]
async fn pipeline_batches_advantages_and_applies_the_policy_update() {
    let policy = Arc::new(RecordingPolicy::default());
    let target = Arc::new(RecordingTarget::new());
    let erased_policy: Arc<dyn Policy> = Arc::clone(&policy);
    let optimizer = Arc::new(OfflinePolicyOptimizer::new(erased_policy));
    assert!(!optimizer.requires_gpu());
    assert_eq!(optimizer.name(), "offline-policy");
    let erased_optimizer: Arc<dyn GradientOptimizer> = Arc::clone(&optimizer);
    let erased_target: Arc<dyn UpdateTarget> = Arc::clone(&target);
    let mut pipeline = RlUpdatePipeline::new(erased_optimizer, erased_target)
    .with_run_id("run-7")
    .with_packer(BatchPacker::new().with_max_samples(2));
    assert_eq!(pipeline.run_id(), "run-7");
    assert_eq!(pipeline.epoch(), 0);
    assert!(!pipeline.requires_gpu());

    let batch = RlBatch::from_samples(
        "batch-1",
        vec![
            sample("s-1", 1.0, Some("g1")),
            sample("s-2", 0.0, Some("g1")),
            sample("s-3", 2.0, Some("g2")),
        ],
    );
    let update = pipeline.apply(&batch).await.expect("apply");

    // Three samples packed into two chunks, then flattened into one gradient batch.
    assert_eq!(update.packed_batches, 2);
    assert_eq!(update.sample_count, 3);
    assert_eq!(update.truncated, 0);
    assert_eq!(update.optimizer, "offline-policy");
    assert_eq!(update.epoch, 1);
    assert_eq!(pipeline.epoch(), 1);
    assert_eq!(update.advantages.advantages, [0.5, -0.5, 0.0]);

    // Batching order: one single-sample policy update per sample, in input order,
    // each carrying its advantage as the reward weight.
    let batches = policy.batches();
    assert_eq!(batches.len(), 3);
    assert_eq!(
        batches
            .iter()
            .map(|batch| batch.batch_id.as_str())
            .collect::<Vec<_>>(),
        ["batch-1-s0", "batch-1-s1", "batch-1-s2"]
    );
    assert_eq!(
        batches
            .iter()
            .map(|batch| batch.samples[0].reward)
            .collect::<Vec<_>>(),
        [0.5, -0.5, 0.0]
    );
    assert_eq!(batches[0].samples[0].metadata["reward"], json!(1.0));
    assert_eq!(batches[0].samples[0].sample_id, "s-1");

    // The single policy update key is written through the target.
    assert_eq!(update.updates.len(), 1);
    assert!(update.updates.contains_key("recording", POLICY_TARGET));
    assert_eq!(update.summary.total, 1);
    assert!(update.summary.is_ok());
    assert_eq!(target.written_targets(), [POLICY_TARGET]);
    assert_eq!(target.value("recording", POLICY_TARGET), Some(policy.parameters()));
    assert_eq!(
        target.value("recording", POLICY_TARGET).expect("written")["updates"],
        json!(3)
    );

    // The checkpoint describes the applied updates and the advantages.
    let checkpoint = &update.checkpoint;
    assert_eq!(checkpoint.version, CHECKPOINT_VERSION);
    assert_eq!(checkpoint.run_id, "run-7");
    assert_eq!(checkpoint.epoch, 1);
    assert_eq!(checkpoint.best_reward, 2.0);
    assert_eq!(checkpoint.metrics["samples"], 3.0);
    assert_eq!(checkpoint.metrics["packed_batches"], 2.0);
    assert_eq!(checkpoint.metrics["applied"], 1.0);
    assert_eq!(checkpoint.metrics["failed"], 0.0);
    assert_eq!(checkpoint.policy["optimizer"], json!("offline-policy"));
    assert_eq!(checkpoint.policy["advantages"]["advantages"], json!([0.5, -0.5, 0.0]));
    assert_eq!(checkpoint.policy["updates"][0]["operator_id"], json!("recording"));
    assert_eq!(checkpoint.file_name(), "epoch-1.json");

    // Persistence is a separate, explicit step.
    let dir = TempDir::new().expect("temp dir");
    let path = RlUpdatePipeline::save_checkpoint(&update, dir.path())
        .await
        .expect("save checkpoint");
    assert!(path.ends_with("epoch-1.json"));
    let loaded = RlCheckpoint::load_from_path(&path).await.expect("load");
    assert_eq!(loaded, *checkpoint);
}

#[tokio::test]
async fn pipeline_reports_none_valued_updates_last_and_fails_loudly() {
    let target = Arc::new(RecordingTarget::new());
    let mut updates = Updates::new();
    updates.insert(("op-b".to_owned(), "body".to_owned()), None);
    updates.insert_update("op-a", "description", json!("applied"));
    let optimizer = Arc::new(ScriptedOptimizer {
        name: "scripted",
        updates,
        gpu: false,
    });
    let erased_optimizer: Arc<dyn GradientOptimizer> = optimizer;
    let erased_target: Arc<dyn UpdateTarget> = Arc::clone(&target);
    let mut pipeline = RlUpdatePipeline::new(erased_optimizer, erased_target);
    let batch = RlBatch::from_samples("batch-2", vec![sample("s-1", 0.5, Some("g"))]);
    let error = pipeline.apply(&batch).await.expect_err("a None update fails");
    assert_eq!(error.kind(), ErrorKind::Workflow);
    assert!(error.to_string().contains("1 of 2 update(s) failed"));
    assert!(error.to_string().contains("update value is None"));
    // The non-None update still reached the target; nothing was silently dropped.
    assert_eq!(target.written_targets(), ["description"]);
    assert_eq!(pipeline.epoch(), 0, "a failed step does not advance the epoch");
}

#[tokio::test]
async fn pipeline_rejects_an_empty_batch() {
    let target = Arc::new(RecordingTarget::new());
    let optimizer = Arc::new(ScriptedOptimizer {
        name: "scripted",
        updates: Updates::new(),
        gpu: false,
    });
    let erased_optimizer: Arc<dyn GradientOptimizer> = optimizer;
    let erased_target: Arc<dyn UpdateTarget> = Arc::clone(&target);
    let mut pipeline = RlUpdatePipeline::new(erased_optimizer, erased_target);
    let error = pipeline
        .apply(&RlBatch::from_samples("batch-3", Vec::new()))
        .await
        .expect_err("empty batch");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("carries no samples"));
}

#[tokio::test]
async fn gpu_gradient_optimizer_reports_the_typed_unsupported_error() {
    let target = Arc::new(RecordingTarget::new());
    let optimizer = Arc::new(GpuGradientOptimizer);
    assert!(optimizer.requires_gpu());
    assert_eq!(optimizer.name(), "gpu");
    let erased_optimizer: Arc<dyn GradientOptimizer> = optimizer;
    let erased_target: Arc<dyn UpdateTarget> = Arc::clone(&target);
    let mut pipeline = RlUpdatePipeline::new(erased_optimizer, erased_target);
    assert!(pipeline.requires_gpu());
    let batch = RlBatch::from_samples("batch-4", vec![sample("s-1", 1.0, Some("g"))]);
    let error = pipeline.apply(&batch).await.expect_err("gpu is deferred");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    let message = error.to_string();
    assert!(message.contains("GpuGradientOptimizer::gradients"));
    for dependency in ["torch", "verl", "ray"] {
        assert!(message.contains(dependency), "message must name {dependency}");
    }
    assert!(message.contains("docs/PARITY.md"));
    assert!(target.is_empty());
    assert_eq!(pipeline.epoch(), 0);
}

#[tokio::test]
async fn offline_optimizer_rejects_a_mismatched_advantage_report() {
    let policy = Arc::new(RecordingPolicy::default());
    let erased: Arc<dyn Policy> = Arc::clone(&policy);
    let optimizer = OfflinePolicyOptimizer::new(erased);
    let batch = RlBatch::from_samples("batch-5", vec![sample("s-1", 1.0, None)]);
    let error = optimizer
        .gradients(&batch, &AdvantageReport::default())
        .await
        .expect_err("mismatched advantages");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("1 sample"));
    assert!(policy.batches().is_empty());
}
