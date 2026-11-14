//! Behaviour tests for the deferred ML/GPU-backed stubs in
//! [`openjiuwen_symphony::deferred`].
//!
//! Every stub must fail the same way: an [`ErrorKind::Unsupported`] error from
//! `openjiuwen_symphony::error::unsupported`, recognized by
//! [`openjiuwen_symphony::error::is_unsupported`], whose message names the deferred
//! feature (so a caller can log it) and points at the crate's deferred scope (so a
//! reader can find the justification).
#![allow(clippy::unwrap_used, clippy::expect_used)]
use openjiuwen_retrieval::HashEmbedding;
use openjiuwen_symphony::deferred::VllmLogitSelector;
use openjiuwen_symphony::error::{is_unsupported, Error, ErrorKind};
use openjiuwen_symphony::experience::ExperienceIndex;

/// Asserts the canonical deferred-stub failure shape for `feature`.
fn assert_unsupported(result: Result<impl std::fmt::Debug, Error>, feature: &str) {
    let error = result.expect_err("deferred stub must not succeed");
    assert_eq!(error.kind(), ErrorKind::Unsupported, "{feature}");
    assert!(is_unsupported(&error), "{feature}");
    let message = error.message();
    assert!(message.contains(feature), "{feature}: {message}");
    assert!(message.contains("deferred"), "{feature}: {message}");
}

#[tokio::test]
async fn faiss_ivf_training_is_deferred() {
    let dir = tempfile::tempdir().expect("tempdir");
    let index = ExperienceIndex::open_with_algorithm(
        dir.path(),
        std::sync::Arc::new(HashEmbedding::new(8)),
        "ivf",
    )
    .await;
    assert_unsupported(index, "faiss IVF");
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
