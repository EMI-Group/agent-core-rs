//! Behaviour tests for the deferred ML/GPU-backed stubs in
//! [`openjiuwen_symphony::deferred`].
//!
//! Every stub must fail the same way: an [`ErrorKind::Unsupported`] error from
//! `openjiuwen_symphony::error::unsupported`, recognized by
//! [`openjiuwen_symphony::error::is_unsupported`], whose message names the deferred
//! feature (so a caller can log it) and points at the crate's deferred scope (so a
//! reader can find the justification).
#![allow(clippy::unwrap_used, clippy::expect_used)]
use openjiuwen_symphony::deferred::{FaissExperienceIndex, VllmLogitSelector};
use openjiuwen_symphony::error::{is_unsupported, Error, ErrorKind};

/// Asserts the canonical deferred-stub failure shape for `feature`.
fn assert_unsupported(result: Result<impl std::fmt::Debug, Error>, feature: &str) {
    let error = result.expect_err("deferred stub must not succeed");
    assert_eq!(error.kind(), ErrorKind::Unsupported, "{feature}");
    assert!(is_unsupported(&error), "{feature}");
    let message = error.message();
    assert!(message.contains(feature), "{feature}: {message}");
    assert!(message.contains("deferred"), "{feature}: {message}");
}

#[test]
fn faiss_experience_index_is_deferred() {
    let root = std::env::temp_dir();
    assert_unsupported(FaissExperienceIndex::new(&root), "faiss experience index");
    assert_unsupported(
        FaissExperienceIndex::search_by_embedding("tokenizer", 5, 0.7),
        "faiss experience index",
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
