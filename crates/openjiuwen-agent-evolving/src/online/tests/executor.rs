//! Unit tests of the executor seam ([`super::super::executor`]).
//!
//! The offline executor is proven against a temp directory; the GPU executor is
//! proven to fail with a typed `Unsupported` that names every missing piece —
//! never to fake training.

use std::collections::BTreeMap;

use serde_json::json;

use super::super::executor::{
    artifact_name, artifact_version, is_artifact_name, LocalArtifactDescriptor,
    LocalTrainingExecutor, PpoTrainingExecutor, TrainingArtifact, TrainingExecutor,
    TrainingRequest,
};
use super::super::sample::SampleRecord;
use crate::rl::RlTrainingConfig;

fn sample(sample_id: &str, reward: f64) -> SampleRecord {
    SampleRecord::pending(sample_id, "owner", reward, json!({}))
}

fn request(samples: Vec<SampleRecord>) -> TrainingRequest {
    TrainingRequest {
        training_run_id: "run-000000000001".to_owned(),
        model_id: "model-a".to_owned(),
        samples,
        init_lora_name: Some("base".to_owned()),
        init_lora_path: Some(String::new()),
        output_dir: None,
        config: RlTrainingConfig::default(),
    }
}

// ------------------------------------------------------------ name contract

#[test]
fn artifact_names_follow_the_model_id_version_contract() {
    assert_eq!(artifact_name("model-a", 3), "model-a:v3");
    for valid in ["model-a:v1", "model-a:v10", "model-a:v1234"] {
        assert!(is_artifact_name("model-a", valid), "{valid}");
        assert!(TrainingArtifact::new(valid, "/tmp/x")
            .validate_name("model-a")
            .is_ok());
    }
    for invalid in [
        "model-a:v0",
        "model-a:v01",
        "model-a:v",
        "model-a:1",
        "model-a:v1x",
        "model-a:v1:extra",
        "model-b:v1",
        "model-a",
        "",
    ] {
        assert!(!is_artifact_name("model-a", invalid), "{invalid}");
    }
    assert_eq!(artifact_version("model-a", "model-a:v7"), Some(7));
    assert_eq!(artifact_version("model-a", "model-a:v0"), None);
    assert!(
        !is_artifact_name("", ":v1"),
        "an empty model has no artifact"
    );
}

#[test]
fn a_foreign_artifact_name_is_a_validation_error() {
    let artifact = TrainingArtifact::new("model-b:v2", "/tmp/x");
    let error = artifact.validate_name("model-a").unwrap_err();
    assert!(matches!(
        error,
        openjiuwen_common::error::Error::Validation(ref message)
            if message == "PPO artifact must use '<model_id>:vN' name"
    ));

    // A model id containing regex metacharacters is compared literally.
    assert!(is_artifact_name("model.a", "model.a:v2"));
    assert!(!is_artifact_name("model.a", "modelXa:v2"));
}

// ------------------------------------------------------- local executor

#[tokio::test]
async fn local_executor_writes_the_descriptor_and_reports_metrics() {
    let temp = tempfile::tempdir().expect("tempdir");
    let executor = LocalTrainingExecutor::new(temp.path());
    let artifact = executor
        .train_batch(&request(vec![
            sample("s1", 0.2),
            sample("s2", 0.8),
            sample("s3", 0.5),
        ]))
        .await
        .expect("train");

    assert_eq!(artifact.lora_name, "model-a:v1");
    assert_eq!(
        artifact.lora_path,
        temp.path()
            .join("run-000000000001")
            .join("v1.json")
            .to_string_lossy()
    );
    assert_eq!(artifact.metrics.get("sample_count"), Some(&3.0));
    assert!((artifact.metrics["mean_reward"] - 0.5).abs() < 1e-12);
    assert!((artifact.metrics["best_reward"] - 0.8).abs() < 1e-12);
    assert_eq!(artifact.metrics.get("policy_version"), Some(&1.0));

    let encoded = tokio::fs::read_to_string(&artifact.lora_path)
        .await
        .expect("descriptor");
    let descriptor: LocalArtifactDescriptor = serde_json::from_str(&encoded).expect("decode");
    assert_eq!(descriptor.lora_name, "model-a:v1");
    assert_eq!(descriptor.model_id, "model-a");
    assert_eq!(descriptor.version, 1);
    assert_eq!(descriptor.sample_count, 3);
    assert!((descriptor.mean_reward - 0.5).abs() < 1e-12);
    assert!((descriptor.best_reward - 0.8).abs() < 1e-12);
    assert_eq!(descriptor.base, "base");

    // The descriptor is the artifact: both agree on every field the artifact carries.
    let replayed = TrainingArtifact {
        lora_name: descriptor.lora_name.clone(),
        lora_path: artifact.lora_path.clone(),
        metrics: BTreeMap::new(),
    };
    assert!(replayed.validate_name("model-a").is_ok());
}

#[tokio::test]
async fn local_executor_bumps_the_policy_version_of_its_parent() {
    let temp = tempfile::tempdir().expect("tempdir");
    let executor = LocalTrainingExecutor::new(temp.path());
    let mut next = request(vec![sample("s1", 1.0)]);
    next.init_lora_name = Some("model-a:v4".to_owned());
    let artifact = executor.train_batch(&next).await.expect("train");
    assert_eq!(artifact.lora_name, "model-a:v5");
    assert_eq!(artifact.metrics.get("policy_version"), Some(&5.0));

    // A parent that is not this model's artifact starts the version at v1.
    let mut foreign_parent = request(vec![sample("s1", 1.0)]);
    foreign_parent.init_lora_name = Some("other-model:v9".to_owned());
    let artifact = executor.train_batch(&foreign_parent).await.expect("train");
    assert_eq!(artifact.lora_name, "model-a:v1");
}

#[tokio::test]
async fn local_executor_uses_the_request_output_dir_and_rejects_bad_batches() {
    let temp = tempfile::tempdir().expect("tempdir");
    let executor = LocalTrainingExecutor::new(temp.path().join("unused"));
    let mut configured = request(vec![sample("s1", 0.4)]);
    configured.output_dir = Some(temp.path().to_path_buf());
    let artifact = executor.train_batch(&configured).await.expect("train");
    assert!(artifact.lora_path.starts_with(
        temp.path()
            .join("run-000000000001")
            .to_string_lossy()
            .as_ref()
    ));

    let empty = executor
        .train_batch(&request(Vec::new()))
        .await
        .unwrap_err();
    assert!(matches!(
        empty,
        openjiuwen_common::error::Error::Validation(ref message)
            if message == "training batch must contain at least one sample"
    ));

    let mut blank_model = request(vec![sample("s1", 0.4)]);
    blank_model.model_id = " ".to_owned();
    let error = executor.train_batch(&blank_model).await.unwrap_err();
    assert!(matches!(
        error,
        openjiuwen_common::error::Error::Validation(ref message) if message == "model_id is required"
    ));

    let nan = executor
        .train_batch(&request(vec![sample("s1", f64::NAN)]))
        .await
        .unwrap_err();
    assert!(matches!(
        nan,
        openjiuwen_common::error::Error::Validation(ref message)
            if message.contains("reward must be finite")
    ));
}

#[tokio::test]
async fn local_executor_reports_health_and_has_nothing_to_cancel() {
    let temp = tempfile::tempdir().expect("tempdir");
    let executor = LocalTrainingExecutor::new(temp.path());
    assert_eq!(executor.name(), "local");
    executor.check_health().await.expect("healthy");
    assert!(!executor.cancel("run-000000000001").await.expect("cancel"));
}

// ---------------------------------------------------------- PPO executor

#[tokio::test]
async fn ppo_executor_is_unsupported_and_names_every_missing_piece() {
    let executor = PpoTrainingExecutor::new();
    assert_eq!(executor.name(), "ppo");
    assert_eq!(executor.device(), "cuda:0");

    for error in [
        executor
            .train_batch(&request(vec![sample("s1", 0.1)]))
            .await
            .unwrap_err(),
        executor.cancel("run-000000000001").await.unwrap_err(),
        executor.check_health().await.unwrap_err(),
    ] {
        assert!(
            matches!(error, openjiuwen_common::error::Error::Unsupported(_)),
            "unexpected variant: {error:?}"
        );
        let message = error.to_string();
        for needle in ["torch", "transformers", "verl", "ray", "CUDA device cuda:0"] {
            assert!(message.contains(needle), "{needle} missing from {message}");
        }
    }

    let on_device = PpoTrainingExecutor::with_device("cuda:3");
    assert_eq!(on_device.device(), "cuda:3");
    assert!(on_device
        .check_health()
        .await
        .unwrap_err()
        .to_string()
        .contains("cuda:3"));
    assert_eq!(PpoTrainingExecutor::default(), PpoTrainingExecutor::new());
}
