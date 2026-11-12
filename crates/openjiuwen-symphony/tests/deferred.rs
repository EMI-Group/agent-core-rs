//! Behaviour tests for the deferred Python-parity stubs in
//! [`openjiuwen_symphony::deferred`].
//!
//! Every stub must fail the same way: an [`ErrorKind::Unsupported`] error from
//! `openjiuwen_symphony::error::unsupported`, recognized by
//! [`openjiuwen_symphony::error::is_unsupported`], whose message names the deferred
//! feature (so a caller can log it) and points at the crate's deferred scope (so a
//! reader can find the justification). The request/result data types stay usable:
//! they round-trip through JSON with `snake_case` keys and tolerate unknown fields.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::float_cmp)]
use openjiuwen_symphony::deferred::{
    ExperienceBank, ExperienceRecord, FlowEngine, FlowReceipt, LlmTreeRetriever,
    LlmTreeRetrieverConfig, SkillFolderScanner, TreeSearchRequest, TreeSearchResult,
};
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
fn llm_tree_retriever_is_deferred() {
    assert_unsupported(
        LlmTreeRetriever::new(LlmTreeRetrieverConfig::default()),
        "LLM tree retrieval",
    );
    assert_unsupported(
        LlmTreeRetriever::search("translate this", &TreeSearchRequest { top_k: Some(3) }),
        "LLM tree retrieval",
    );
}

#[test]
fn flow_engine_is_deferred() {
    assert_unsupported(FlowEngine::new(), "flow distillation");
    assert_unsupported(FlowEngine::distill(), "flow distillation");
    assert_unsupported(FlowEngine::pack(), "flow distillation");
}

#[test]
fn experience_bank_is_deferred() {
    let record = ExperienceRecord {
        text: "cache the tokenizer between calls".to_string(),
        score: 0.91,
    };
    assert_unsupported(ExperienceBank::new(), "experience bank");
    assert_unsupported(ExperienceBank::add(record), "experience bank");
    assert_unsupported(ExperienceBank::search("tokenizer", 5), "experience bank");
}

#[test]
fn skill_folder_scanner_is_deferred() {
    let root = std::env::temp_dir();
    assert_unsupported(SkillFolderScanner::new(&root), "skill folder scanning");
    assert_unsupported(SkillFolderScanner::scan(&root), "skill folder scanning");
}

#[test]
fn retriever_config_round_trips_and_defaults_like_python() {
    let config = LlmTreeRetrieverConfig::default();
    assert_eq!(config.top_k, 10);
    assert_eq!(config.model, None);
    let decoded: LlmTreeRetrieverConfig =
        serde_json::from_str(r#"{"top_k": 4, "model": "qwen", "unknown": {"x": 1}}"#).unwrap();
    assert_eq!(decoded.top_k, 4);
    assert_eq!(decoded.model.as_deref(), Some("qwen"));
    let encoded = serde_json::to_value(&decoded).unwrap();
    assert_eq!(encoded, serde_json::json!({"top_k": 4, "model": "qwen"}));
    let empty: LlmTreeRetrieverConfig = serde_json::from_str("{}").unwrap();
    assert_eq!(empty, LlmTreeRetrieverConfig::default());
}

#[test]
fn tree_search_request_and_result_round_trip() {
    let request = TreeSearchRequest::default();
    assert_eq!(request.top_k, None);
    let decoded: TreeSearchRequest = serde_json::from_str(r#"{"top_k": 7}"#).unwrap();
    assert_eq!(decoded.top_k, Some(7));

    let result = TreeSearchResult {
        payloads: vec!["skill-a".to_string(), "skill-b".to_string()],
        method: "progressive".to_string(),
        elapsed_ms: 42,
    };
    let encoded = serde_json::to_value(&result).unwrap();
    assert_eq!(
        encoded,
        serde_json::json!({
            "payloads": ["skill-a", "skill-b"],
            "method": "progressive",
            "elapsed_ms": 42,
        })
    );
    let decoded: TreeSearchResult = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded, result);
    assert_eq!(TreeSearchResult::default().elapsed_ms, 0);
}

#[test]
fn flow_receipt_and_experience_record_round_trip() {
    let receipt = FlowReceipt {
        revision: "v3".to_string(),
    };
    let encoded = serde_json::to_value(&receipt).unwrap();
    assert_eq!(encoded, serde_json::json!({"revision": "v3"}));
    let decoded: FlowReceipt = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded, receipt);
    assert_eq!(FlowReceipt::default().revision, "");

    let record = ExperienceRecord {
        text: "prefer bfs over dfs for shortest path".to_string(),
        score: 0.5,
    };
    let encoded = serde_json::to_value(&record).unwrap();
    assert_eq!(
        encoded,
        serde_json::json!({"text": "prefer bfs over dfs for shortest path", "score": 0.5})
    );
    let decoded: ExperienceRecord = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded, record);
    let empty: ExperienceRecord = serde_json::from_str("{}").unwrap();
    assert_eq!(empty, ExperienceRecord::default());
}
