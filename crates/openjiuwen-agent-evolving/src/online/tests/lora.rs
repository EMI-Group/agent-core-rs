//! Unit tests of the LoRA artifact repository ([`super::super::lora`]).
//!
//! The repository is filesystem-backed, so every test publishes into its own
//! temp directory and reads the tree back — including through a second repository
//! instance, which is what a restarted service does.

use super::super::executor::TrainingArtifact;
use super::super::lora::{
    parse_version_literal, version_literal, LoraAvailability, LoraRepository,
};

fn artifact(model_id: &str, version: u32) -> TrainingArtifact {
    let mut artifact = TrainingArtifact::new(
        format!("{model_id}:v{version}"),
        format!("/artifacts/{model_id}/v{version}"),
    );
    artifact.metrics.insert("sample_count".to_owned(), 4.0);
    artifact
}

#[tokio::test]
async fn publish_allocates_increasing_versions_and_updates_latest() {
    let temp = tempfile::tempdir().expect("tempdir");
    let repo = LoraRepository::new(temp.path());

    let first = repo
        .publish("model-a", &artifact("model-a", 1))
        .await
        .expect("publish v1");
    assert_eq!(first.model_id, "model-a");
    assert_eq!(first.version, 1);
    assert_eq!(first.version_literal(), "v1");
    assert_eq!(first.lora_name, "model-a:v1");
    assert_eq!(first.lora_path, "/artifacts/model-a/v1");
    assert_eq!(first.availability, LoraAvailability::Pending);
    assert_eq!(first.reason, "");
    assert_eq!(first.created_seq, 1);

    let second = repo
        .publish("model-a", &artifact("model-a", 2))
        .await
        .expect("publish v2");
    assert_eq!(second.version, 2);
    assert_eq!(second.created_seq, 2);

    let versions = repo.list_versions("model-a").await.expect("versions");
    assert_eq!(
        versions.iter().map(|v| v.version).collect::<Vec<_>>(),
        [1, 2]
    );
    assert_eq!(
        repo.get_latest("model-a")
            .await
            .expect("latest")
            .expect("some")
            .version,
        2
    );
    assert_eq!(repo.list_models().await.expect("models"), ["model-a"]);

    // The descriptor and the version record live in the version directory.
    let stored = repo
        .artifact("model-a", 2)
        .await
        .expect("artifact")
        .expect("descriptor");
    assert_eq!(stored, artifact("model-a", 2));
    let metadata =
        tokio::fs::read_to_string(temp.path().join("model-a").join("v2").join("version.json"))
            .await
            .expect("version.json");
    assert!(metadata.contains("\"lora_name\": \"model-a:v2\""));
    let latest = tokio::fs::read_to_string(temp.path().join("model-a").join("latest"))
        .await
        .expect("latest pointer");
    assert_eq!(latest, "v2");
}

#[tokio::test]
async fn availability_follows_the_inference_probe() {
    let temp = tempfile::tempdir().expect("tempdir");
    let repo = LoraRepository::new(temp.path());
    repo.publish("model-a", &artifact("model-a", 1))
        .await
        .expect("v1");
    repo.publish("model-a", &artifact("model-a", 2))
        .await
        .expect("v2");

    assert_eq!(
        repo.get_latest_available("model-a")
            .await
            .expect("available"),
        None,
        "everything starts pending"
    );

    let available = repo
        .set_availability("model-a", 1, true, "served by the gateway")
        .await
        .expect("available v1");
    assert_eq!(available.availability, LoraAvailability::Available);
    assert_eq!(available.reason, "served by the gateway");
    assert_eq!(
        repo.get_latest_available("model-a")
            .await
            .expect("available")
            .expect("v1")
            .version,
        1
    );

    repo.set_availability("model-a", 2, true, "served")
        .await
        .expect("available v2");
    assert_eq!(
        repo.get_latest_available("model-a")
            .await
            .expect("available")
            .expect("v2")
            .version,
        2,
        "the newest available version wins"
    );

    let rejected = repo
        .set_availability("model-a", 2, false, "weight load failed")
        .await
        .expect("unavailable v2");
    assert_eq!(rejected.availability, LoraAvailability::Unavailable);
    assert_eq!(rejected.reason, "weight load failed");
    assert_eq!(
        repo.get_latest_available("model-a")
            .await
            .expect("available")
            .expect("v1")
            .version,
        1,
        "an unavailable version falls back to the previous one"
    );
    // `latest` is untouched by availability.
    assert_eq!(
        repo.get_latest("model-a")
            .await
            .expect("latest")
            .expect("some")
            .version,
        2
    );

    let error = repo
        .set_availability("model-a", 9, true, "missing")
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        openjiuwen_common::error::Error::NotFound(ref message)
            if message == "LoRA version not found: model-a/v9"
    ));
}

#[tokio::test]
async fn the_repository_is_durable_across_instances() {
    let temp = tempfile::tempdir().expect("tempdir");
    let first = LoraRepository::new(temp.path());
    first
        .publish("model-a", &artifact("model-a", 1))
        .await
        .expect("v1");
    first
        .set_availability("model-a", 1, true, "served")
        .await
        .expect("available");

    let reopened = LoraRepository::new(temp.path());
    let latest = reopened
        .get_latest("model-a")
        .await
        .expect("latest")
        .expect("v1");
    assert_eq!(latest.version, 1);
    assert_eq!(latest.availability, LoraAvailability::Available);
    assert_eq!(latest.reason, "served");
    assert_eq!(
        reopened
            .get_version("model-a", 1)
            .await
            .expect("version")
            .expect("some"),
        latest
    );
    // A new publish appends after the persisted versions.
    let next = reopened
        .publish("model-a", &artifact("model-a", 2))
        .await
        .expect("v2");
    assert_eq!(next.version, 2);
    assert_eq!(
        reopened.list_versions("model-a").await.expect("list").len(),
        2
    );
}

#[tokio::test]
async fn missing_models_and_versions_read_as_empty() {
    let temp = tempfile::tempdir().expect("tempdir");
    let repo = LoraRepository::new(temp.path().join("missing"));
    assert!(repo
        .list_versions("model-a")
        .await
        .expect("list")
        .is_empty());
    assert!(repo.get_latest("model-a").await.expect("latest").is_none());
    assert!(repo
        .get_latest_available("model-a")
        .await
        .expect("available")
        .is_none());
    assert!(repo
        .get_version("model-a", 1)
        .await
        .expect("version")
        .is_none());
    assert!(repo
        .artifact("model-a", 1)
        .await
        .expect("artifact")
        .is_none());
    assert!(repo.list_models().await.expect("models").is_empty());
    // A directory that exists but holds no version is still a model with none.
    tokio::fs::create_dir_all(temp.path().join("model-a").join("notes"))
        .await
        .expect("dir");
    assert!(repo
        .list_versions("model-a")
        .await
        .expect("list")
        .is_empty());
    assert!(repo.get_latest("model-a").await.expect("latest").is_none());
}

#[tokio::test]
async fn publish_rejects_foreign_names_and_unsafe_model_ids() {
    let temp = tempfile::tempdir().expect("tempdir");
    let repo = LoraRepository::new(temp.path());

    let error = repo
        .publish("model-a", &artifact("model-b", 1))
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        openjiuwen_common::error::Error::Validation(ref message)
            if message == "PPO artifact must use '<model_id>:vN' name"
    ));

    for model_id in ["", "  ", "..", "../escape", "nested/path", ".hidden"] {
        let error = repo
            .publish(model_id, &artifact("model-a", 1))
            .await
            .unwrap_err();
        assert!(
            matches!(error, openjiuwen_common::error::Error::Validation(_)),
            "{model_id} produced {error:?}"
        );
    }
}

#[tokio::test]
async fn models_are_listed_sorted_and_hidden_entries_are_skipped() {
    let temp = tempfile::tempdir().expect("tempdir");
    let repo = LoraRepository::new(temp.path());
    for model in ["model-b", "model-a"] {
        repo.publish(model, &artifact(model, 1))
            .await
            .expect("publish");
    }
    tokio::fs::create_dir_all(temp.path().join(".scratch"))
        .await
        .expect("hidden dir");
    tokio::fs::write(temp.path().join("stray.txt"), "x")
        .await
        .expect("stray file");
    assert_eq!(
        repo.list_models().await.expect("models"),
        ["model-a", "model-b"]
    );
}

#[test]
fn version_literals_parse_strictly() {
    assert_eq!(version_literal(3), "v3");
    assert_eq!(parse_version_literal("v3"), Some(3));
    assert_eq!(parse_version_literal("v10"), Some(10));
    for invalid in ["v0", "v01", "v", "3", "latest", ".v1", "v1x"] {
        assert_eq!(parse_version_literal(invalid), None, "{invalid}");
    }
    assert_eq!(LoraAvailability::Pending.as_str(), "pending");
    assert_eq!(LoraAvailability::Available.as_str(), "available");
    assert_eq!(LoraAvailability::Unavailable.as_str(), "unavailable");
}
