//! Behaviour tests for the typed `Error::Unsupported` boundaries of
//! `openjiuwen_symphony`.
//!
//! `deferred::VllmLogitSelector` is the crate's only stub module; the remaining boundaries
//! live with their implementations (the three synchronous grouping strategies behind
//! `IndexBuilder::build`, and `AzureBlobStore::from_client_credentials`). All of them must
//! fail the same way: an [`ErrorKind::Unsupported`] error from
//! `openjiuwen_symphony::error::unsupported`, recognized by
//! [`openjiuwen_symphony::error::is_unsupported`], whose message names the deferred feature
//! (so a caller can log it) and points at the crate's deferred scope (so a reader can find
//! the justification).
//!
//! Two capabilities that sit next to those boundaries are implemented rather than deferred,
//! and are pinned as working: the IVF vector algorithms of
//! [`ExperienceIndex::open_with_algorithm`] (below) and GCS service-account key files (pinned
//! by the `object_artifact::gcs` module suite).
//!
//! Nothing here touches the network: the Azure constructor rejects before any request is
//! built, the IVF index drives a deterministic in-process embedding, and the build-stage
//! boundaries reject before anything is written.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use std::collections::BTreeMap;
use std::sync::Arc;

use openjiuwen_retrieval::HashEmbedding;
use openjiuwen_symphony::deferred::VllmLogitSelector;
use openjiuwen_symphony::error::{is_unsupported, Error, ErrorKind};
use openjiuwen_symphony::experience::{ExperienceIndex, ExperienceItem, ExperienceRanking};
use openjiuwen_symphony::object_artifact::AzureBlobStore;
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
async fn the_ivf_vector_algorithm_opens_and_searches() {
    let dir = tempfile::tempdir().expect("tempdir");
    let index =
        ExperienceIndex::open_with_algorithm(dir.path(), Arc::new(HashEmbedding::new(8)), "ivf")
            .await
            .expect("an IVF index opens on an empty directory");
    assert_eq!(index.algorithm(), "ivf");
    assert_eq!(index.count(), 0);

    index
        .add(ExperienceItem::new("summarize the report"))
        .await
        .expect("add an item");
    assert_eq!(index.count(), 1);
    // One stored vector clamps the requested 64 coarse lists to one, which the default probe
    // covers, so the whole catalog is searched.
    assert_eq!(index.nlist(), 1);

    let hits = index
        .search("summarize the report", &ExperienceRanking::new(1))
        .await
        .expect("search an IVF index");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].item.query_pattern, "summarize the report");
    assert!(dir.path().join("meta.json").exists(), "the index persists");
}

#[tokio::test]
async fn an_unknown_vector_algorithm_is_a_validation_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let error =
        ExperienceIndex::open_with_algorithm(dir.path(), Arc::new(HashEmbedding::new(8)), "hnsw")
            .await
            .expect_err("an unimplemented algorithm must be rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.message();
    assert!(message.contains("vector_algorithm"), "{message}");
    assert!(message.contains("hnsw"), "{message}");
    assert_eq!(
        std::fs::read_dir(dir.path())
            .expect("the output directory")
            .count(),
        0,
        "a rejected open must publish nothing"
    );
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
fn the_azure_client_credential_flow_is_deferred() {
    assert_unsupported(
        AzureBlobStore::from_client_credentials("tenant", "client", "secret"),
        "Azure AD client-credential OAuth",
    );
}
