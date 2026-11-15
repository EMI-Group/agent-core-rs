//! Behaviour tests for every typed `Error::Unsupported` boundary of
//! `openjiuwen_symphony`.
//!
//! `deferred::VllmLogitSelector` is the crate's only stub module; the remaining boundaries
//! live with their implementations
//! (`ExperienceIndex::open_with_algorithm`, the grouping strategies behind
//! `IndexBuilder::build`, and the two object-store credential constructors). All of them
//! must fail the same way: an [`ErrorKind::Unsupported`] error from
//! `openjiuwen_symphony::error::unsupported`, recognized by
//! [`openjiuwen_symphony::error::is_unsupported`], whose message names the deferred
//! feature (so a caller can log it) and points at the crate's deferred scope (so a
//! reader can find the justification).
//!
//! Nothing here touches the network: the credential constructors reject before any request
//! is built, and the two build-stage boundaries reject before anything is written.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use std::collections::BTreeMap;

use openjiuwen_retrieval::HashEmbedding;
use openjiuwen_symphony::deferred::VllmLogitSelector;
use openjiuwen_symphony::error::{is_unsupported, Error, ErrorKind};
use openjiuwen_symphony::experience::ExperienceIndex;
use openjiuwen_symphony::object_artifact::{AzureBlobStore, GcsObjectStore};
use openjiuwen_symphony::retrieval::build::grouping::{GroupingConfig, GroupingStrategy};
use openjiuwen_symphony::retrieval::build::scanner::{ItemSources, SourceItem};
use openjiuwen_symphony::retrieval::build::workflows::{BuildConfig, IndexBuilder};
use openjiuwen_symphony::retrieval::types::RetrieverItem;

/// Asserts the canonical deferred-stub failure shape for `feature`.
///
/// `feature` must be a distinctive phrase of the error message, so the assertion also
/// pins that the message names what is deferred rather than a generic complaint.
fn assert_unsupported(result: Result<impl std::fmt::Debug, Error>, feature: &str) {
    let error = result.expect_err("deferred stub must not succeed");
    assert_eq!(error.kind(), ErrorKind::Unsupported, "{feature}");
    assert!(is_unsupported(&error), "{feature}");
    let message = error.message();
    assert!(message.contains(feature), "{feature}: {message}");
    assert!(message.contains("deferred"), "{feature}: {message}");
    assert!(
        message.contains("Deferred Scope") || message.contains("deferred scope"),
        "{feature} must point at the crate's deferred scope: {message}"
    );
}

/// One scannable skill item, enough for a grouping strategy to be asked to place it.
fn item(worker_id: &str, name: &str, classification: &str) -> SourceItem {
    SourceItem {
        worker_id: worker_id.to_string(),
        item: RetrieverItem::new(worker_id, worker_id),
        name: name.to_string(),
        description: format!("{name} routes matching requests."),
        content: format!("{name} usage notes."),
        skill_path: format!("store/{worker_id}/SKILL.md"),
        classification: classification.to_string(),
        tags: Vec::new(),
        item_type: "skill".to_string(),
        content_hash: String::new(),
        metadata: BTreeMap::new(),
    }
}

/// Two skill items wrapped into one skill source set.
fn sources() -> ItemSources {
    let mut sources = ItemSources::new("skill");
    for entry in [
        item("PdfFill", "PDF Fill", "documents/pdf"),
        item("WebSearch", "Web Search", "web"),
    ] {
        sources
            .item_paths
            .push(format!("store/{}", entry.worker_id));
        sources.items.push(entry);
    }
    sources.sort();
    sources
}

#[tokio::test]
async fn faiss_ivf_training_is_deferred() {
    let dir = tempfile::tempdir().expect("tempdir");
    for algorithm in ["ivf", "ivfpq"] {
        let index = ExperienceIndex::open_with_algorithm(
            dir.path(),
            std::sync::Arc::new(HashEmbedding::new(8)),
            algorithm,
        )
        .await;
        assert_unsupported(index, "faiss IVF");
    }
    assert!(!dir.path().join("meta.json").exists());
}

#[test]
fn vllm_logit_selector_is_deferred() {
    let candidates = vec!["skill-a".to_string(), "skill-b".to_string()];
    assert_unsupported(
        VllmLogitSelector::new("Qwen/Qwen2.5-7B"),
        "vLLM logit selection",
    );
    assert_unsupported(
        VllmLogitSelector::select("translate this", &candidates),
        "vLLM logit selection",
    );
}

#[tokio::test]
async fn the_model_and_faiss_grouping_strategies_are_deferred_builds() {
    let strategies = [
        (GroupingStrategy::FaissKmeans, "faiss k-means tree grouping"),
        (GroupingStrategy::Kmeans, "k-means tree grouping"),
        (GroupingStrategy::LlmGrouping, "LLM-guided tree grouping"),
    ];
    for (strategy, feature) in strategies {
        let dir = tempfile::tempdir().expect("tempdir");
        let config =
            BuildConfig::new().with_grouping(GroupingConfig::default().with_strategy(strategy));
        let report = IndexBuilder::build(sources(), dir.path(), config).await;
        assert_unsupported(report, feature);
        assert_eq!(
            std::fs::read_dir(dir.path())
                .expect("the output directory")
                .count(),
            0,
            "{strategy:?} must not publish an artifact"
        );
    }
}

#[test]
fn the_object_store_credential_flows_are_deferred() {
    let dir = tempfile::tempdir().expect("tempdir");
    let key_file = dir.path().join("service-account.json");
    std::fs::write(&key_file, "{}").expect("the key file is written");
    assert_unsupported(
        GcsObjectStore::from_service_account_file(&key_file),
        "service-account JSON credentials",
    );
    assert_unsupported(
        AzureBlobStore::from_client_credentials("tenant", "client", "secret"),
        "Azure AD client-credential OAuth",
    );
}
