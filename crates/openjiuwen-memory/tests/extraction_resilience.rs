//! Public-surface tests of the `add_messages` extraction bookkeeping.
//!
//! The deterministic `ScriptedModel` drives the model path so the two failure modes that
//! used to be silent are asserted from outside the crate: an unrecognized fact `type` and
//! an empty extraction payload must both keep the extracted facts, and a failed model
//! call must leave an [`AddMemResult::degraded`] reason plus a visible dedupe count.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;

use openjiuwen_foundation::prompt::BaseMessage;
use openjiuwen_memory::config::{AgentMemoryConfig, MemoryEngineConfig};
use openjiuwen_memory::facade::LongTermMemory;
use openjiuwen_memory::record::{AddOptions, DedupePolicy, MemoryScope, MemoryType, NewMemory};
use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};

fn scope() -> MemoryScope {
    MemoryScope::new("alice", "ws")
}

fn turns() -> Vec<BaseMessage> {
    vec![
        BaseMessage::user("I always prefer dark mode in every editor"),
        BaseMessage::assistant("Noted, dark mode it is for your editors"),
    ]
}

fn memory(config: MemoryEngineConfig, model: ScriptedModel) -> Arc<LongTermMemory> {
    LongTermMemory::builder(config)
        .model(Arc::new(model))
        .build()
        .expect("facade")
}

#[tokio::test]
async fn a_failed_model_extraction_reports_a_degradation_reason() {
    let model = ScriptedModel::new([
        ScriptedTurn::fail("provider exploded"),
        ScriptedTurn::fail("provider exploded"),
        ScriptedTurn::fail("provider exploded"),
    ]);
    let ltm = memory(MemoryEngineConfig::default(), model);

    let result = ltm
        .add_messages(
            &scope(),
            &turns(),
            &AgentMemoryConfig::default(),
            &AddOptions::default(),
        )
        .await
        .expect("add messages");

    let degraded = result.degraded.clone().expect("degradation reason");
    assert!(degraded.contains("provider exploded"), "{degraded}");
    assert_eq!(result.episodic_memory.len(), 2);
    assert_eq!(result.summary.len(), 1);
    // The degradation reason survives the public JSON shape.
    let json = serde_json::to_value(&result).expect("json");
    assert_eq!(json["degraded"], serde_json::json!(degraded));
    let stored = ltm.list(&scope(), &[]).await.expect("list");
    assert_eq!(stored.len(), 3, "the heuristic records are persisted");
}

#[tokio::test]
async fn a_model_fact_type_the_crate_does_not_know_is_still_stored() {
    let model = ScriptedModel::text(
        r#"{"facts":[{"content":"Alice prefers tea over coffee","type":"preference"}]}"#,
    );
    let ltm = memory(MemoryEngineConfig::default(), model);

    let result = ltm
        .add_messages(
            &scope(),
            &turns(),
            &AgentMemoryConfig::default(),
            &AddOptions::default(),
        )
        .await
        .expect("add messages");

    assert_eq!(result.degraded, None);
    assert_eq!(result.semantic_memory.len(), 1);
    assert_eq!(result.semantic_memory[0].kind, MemoryType::SemanticMemory);
    assert_eq!(
        result.semantic_memory[0].content,
        "Alice prefers tea over coffee"
    );
    let stored = ltm.list(&scope(), &[]).await.expect("list");
    assert_eq!(stored.len(), 1, "the fact is not silently dropped");
}

#[tokio::test]
async fn an_empty_payload_falls_back_to_the_heuristics() {
    let model = ScriptedModel::text("{}");
    let ltm = memory(MemoryEngineConfig::default(), model);

    let result = ltm
        .add_messages(
            &scope(),
            &turns(),
            &AgentMemoryConfig::default(),
            &AddOptions::default(),
        )
        .await
        .expect("add messages");

    assert_eq!(result.episodic_memory.len(), 2);
    assert_eq!(result.summary.len(), 1);
    assert!(result.degraded.is_some(), "the empty payload is reported");
}

#[tokio::test]
async fn a_skipped_duplicate_shows_up_as_a_counter() {
    let model = ScriptedModel::text(
        r#"{"facts":[{"content":"Alice keeps every editor in dark mode",
                       "type":"semantic_memory"}]}"#,
    );
    let ltm = memory(MemoryEngineConfig::default(), model);
    ltm.add(
        &scope(),
        NewMemory::semantic("Alice keeps every editor in dark mode"),
        &AddOptions::default(),
    )
    .await
    .expect("seed record");

    let result = ltm
        .add_messages(
            &scope(),
            &turns(),
            &AgentMemoryConfig::default(),
            &AddOptions::default().with_dedupe(DedupePolicy::Skip),
        )
        .await
        .expect("add messages");

    assert_eq!(result.skipped, 1);
    assert_eq!(result.semantic_memory.len(), 0);
    assert_eq!(
        ltm.list(&scope(), &[]).await.expect("list").len(),
        1,
        "skipping leaves the stored record untouched"
    );
}
